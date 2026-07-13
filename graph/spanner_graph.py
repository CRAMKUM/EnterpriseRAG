"""Spanner Graph manager for MMKG operations."""

from typing import Dict, Any, List, Optional
from google.cloud import spanner
from google.cloud.spanner_v1 import param_types
from utils.logger import get_logger
from utils.exceptions import GraphError
from config.config_loader import get_config_loader

logger = get_logger(__name__)

class SpannerGraphManager:
    """
    Manages Google Cloud Spanner Graph operations for MMKG.
    """

    def __init__(self):
        """Initialize Spanner Graph connection."""
        self.config_loader = get_config_loader()
        self.graph_config = self.config_loader.get_graph_config()
        self._init_spanner()
        logger.info("SpannerGraphManager initialized")

    def _init_spanner(self):
        """Initialize Spanner client and database connection."""
        try:
            instance_id = self.graph_config.get("instance_id")
            database_id = self.graph_config.get("database_id")
            project_id = self.graph_config.get("project_id")

            if not all([instance_id, database_id, project_id]):
                raise ValueError("Missing required Spanner configuration")

            self.client = spanner.Client(project=project_id)
            self.instance = self.client.instance(instance_id)
            self.database = self.instance.database(database_id)
            logger.info(f"Connected to Spanner: {project_id}/{instance_id}/{database_id}")
        except Exception as e:
            logger.error(f"Failed to initialize Spanner: {e}")
            raise GraphError(f"Spanner initialization failed: {e}")

    def create_schema(self):
        """Create tables for Entities and Relationships."""
        try:
            logger.info("Creating Spanner Graph schema")

            entity_ddl = """
            CREATE TABLE IF NOT EXISTS Entity (
                entity_id STRING(MAX) NOT NULL,
                entity_type STRING(MAX),
                name STRING(MAX),
                description STRING(MAX),
                source_page INT64,
                confidence FLOAT64,
                attributes JSON,
                created_at TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true),
                updated_at TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true)
            ) PRIMARY KEY (entity_id)
            """

            relationship_ddl = """
            CREATE TABLE IF NOT EXISTS Relationship (
                relationship_id STRING(MAX) NOT NULL,
                source_entity_id STRING(MAX) NOT NULL,
                target_entity_id STRING(MAX) NOT NULL,
                relationship_type STRING(MAX),
                description STRING(MAX),
                keywords STRING(MAX),
                strength FLOAT64,
                source_page INT64,
                confidence FLOAT64,
                cross_modal BOOL,
                attributes JSON,
                created_at TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true),
                updated_at TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true),
                FOREIGN KEY (source_entity_id) REFERENCES Entity (entity_id),
                FOREIGN KEY (target_entity_id) REFERENCES Entity (entity_id)
            ) PRIMARY KEY (relationship_id)
            """

            operation = self.database.update_ddl([entity_ddl, relationship_ddl])
            operation.result()
            logger.info("Schema created successfully")
        except Exception as e:
            logger.error(f"Schema creation failed: {e}")
            raise GraphError(f"Schema creation failed: {e}")

    def insert_entities(self, entities: List[Dict[str, Any]]) -> int:
        """Insert multiple entities into Spanner."""
        try:
            logger.info(f"Inserting {len(entities)} entities")

            def insert_batch(transaction):
                count = 0
                for entity in entities:
                    transaction.insert(
                        table="Entity",
                        columns=["entity_id", "entity_type", "name", "description",
                                 "source_page", "confidence", "attributes", "created_at", "updated_at"],
                        values=[[
                            entity.get("id"),
                            entity.get("type"),
                            entity.get("name"),
                            entity.get("description"),
                            entity.get("source_page"),
                            entity.get("confidence", 0.8),
                            spanner.Json(entity.get("attributes", {})),
                            spanner.COMMIT_TIMESTAMP,
                            spanner.COMMIT_TIMESTAMP
                        ]]
                    )
                    count += 1
                return count

            inserted = self.database.run_in_transaction(insert_batch)
            logger.info(f"Inserted {inserted} entities")
            return inserted
        except Exception as e:
            logger.error(f"Entity insertion failed: {e}")
            raise GraphError(f"Entity insertion failed: {e}")

    def insert_relationships(self, relationships: List[Dict[str, Any]]) -> int:
        """Insert relationships."""
        try:
            logger.info(f"Inserting {len(relationships)} relationships")

            def insert_batch(transaction):
                count = 0
                for rel in relationships:
                    rel_id = f"{rel.get('source_entity')}_{rel.get('target_entity')}_{count}"
                    transaction.insert(
                        table="Relationship",
                        columns=["relationship_id", "source_entity_id", "target_entity_id",
                                 "relationship_type", "description", "keywords", "strength",
                                 "source_page", "confidence", "cross_modal", "attributes", "created_at", "updated_at"],
                        values=[[
                            rel_id,
                            rel.get("source_entity") or rel.get("source_id"),
                            rel.get("target_entity") or rel.get("target_id"),
                            rel.get("type"),
                            rel.get("description"),
                            rel.get("keywords"),
                            rel.get("strength"),
                            rel.get("source_page"),
                            rel.get("confidence", 0.8),
                            rel.get("cross_modal", False),
                            spanner.Json(rel.get("attributes", {})),
                            spanner.COMMIT_TIMESTAMP,
                            spanner.COMMIT_TIMESTAMP
                        ]]
                    )
                    count += 1
                return count

            inserted = self.database.run_in_transaction(insert_batch)
            logger.info(f"Inserted {inserted} relationships")
            return inserted
        except Exception as e:
            logger.error(f"Relationship insertion failed: {e}")
            raise GraphError(f"Relationship insertion failed: {e}")

    def execute_gql(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute SQL/GQL query on database."""
        try:
            logger.info(f"Executing query: {query[:100]}...")
            with self.database.snapshot() as snapshot:
                if params:
                    param_values = {}
                    param_type_dict = {}
                    for key, value in params.items():
                        param_values[key] = value
                        if isinstance(value, str):
                            param_type_dict[key] = param_types.STRING
                        elif isinstance(value, int):
                            param_type_dict[key] = param_types.INT64
                        elif isinstance(value, float):
                            param_type_dict[key] = param_types.FLOAT64
                    results = snapshot.execute_sql(
                        query,
                        params=param_values,
                        param_types=param_type_dict
                    )
                else:
                    results = snapshot.execute_sql(query)

            rows = []
            for row in results:
                rows.append(dict(zip(results.fields, row)))
            return rows
        except Exception as e:
            logger.error(f"GQL query execution failed: {e}")
            raise GraphError(f"GQL query failed: {e}")

    def get_entity_by_id(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get single entity by ID."""
        try:
            query = "SELECT * FROM Entity WHERE entity_id = @entity_id"
            results = self.execute_gql(query, {"entity_id": entity_id})
            return results[0] if results else None
        except Exception as e:
            logger.error(f"Failed to get entity {entity_id}: {e}")
            return None

    def get_entity_relationships(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get all relationships connected to entity."""
        try:
            query = """
                SELECT * FROM Relationship
                WHERE source_entity_id = @entity_id
                OR target_entity_id = @entity_id
            """
            return self.execute_gql(query, {"entity_id": entity_id})
        except Exception as e:
            logger.error(f"Failed to get relationships for {entity_id}: {e}")
            return []

    def search_entities_by_name(self, name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search entities by name (case-insensitive fuzzy search)."""
        try:
            query = f"""
                SELECT * FROM Entity
                WHERE LOWER(name) LIKE @name_pattern
                LIMIT {limit}
            """
            name_pattern = f"%{name.lower()}%"
            return self.execute_gql(query, {"name_pattern": name_pattern})
        except Exception as e:
            logger.error(f"Entity search failed: {e}")
            return []

    def update_entity(self, entity_id: str, updates: Dict[str, Any]) -> bool:
        """Update entity attributes."""
        try:
            logger.info(f"Updating entity {entity_id}")

            def update_transaction(transaction):
                update_fields = []
                for key, value in updates.items():
                    if key != "entity_id":
                        update_fields.append(f"{key} = @{key}")

                if not update_fields:
                    return False

                query = f"""
                    UPDATE Entity
                    SET {', '.join(update_fields)}, updated_at = PENDING_COMMIT_TIMESTAMP()
                    WHERE entity_id = @entity_id
                """
                params = {**updates, "entity_id": entity_id}
                transaction.execute_update(query, params=params)
                return True

            self.database.run_in_transaction(update_transaction)
            logger.info(f"Updated entity {entity_id}")
            return True
        except Exception as e:
            logger.error(f"Entity update failed: {e}")
            return False

    def delete_relationship(self, relationship_id: str) -> bool:
        """Delete relationship from database."""
        try:
            logger.info(f"Deleting relationship {relationship_id}")

            def delete_transaction(transaction):
                transaction.execute_update(
                    "DELETE FROM Relationship WHERE relationship_id = @rel_id",
                    params={"rel_id": relationship_id}
                )

            self.database.run_in_transaction(delete_transaction)
            logger.info(f"Deleted relationship {relationship_id}")
            return True
        except Exception as e:
            logger.error(f"Relationship deletion failed: {e}")
            return False

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        try:
            entity_count = self.execute_gql("SELECT COUNT(*) as count FROM Entity")[0]["count"]
            rel_count = self.execute_gql("SELECT COUNT(*) as count FROM Relationship")[0]["count"]

            entity_types = self.execute_gql("""
                SELECT entity_type, COUNT(*) as count
                FROM Entity
                GROUP BY entity_type
            """)

            return {
                "total_entities": entity_count,
                "total_relationships": rel_count,
                "entity_types": entity_types,
                "success": True
            }
        except Exception as e:
            logger.error(f"Failed to get graph statistics: {e}")
            return {"success": False, "error": str(e)}

    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 3
    ) -> List[Dict[str, Any]]:
        """Find paths between source and target entities."""
        try:
            logger.info(f"Finding paths from {source_id} to {target_id}")

            query = f"""
            WITH RECURSIVE path AS (
                SELECT
                    entity_id,
                    name,
                    entity_type,
                    ARRAY[entity_id] as path_ids,
                    0 as depth
                FROM Entity
                WHERE entity_id = @source_id

                UNION ALL

                SELECT
                    e.entity_id,
                    e.name,
                    e.entity_type,
                    ARRAY_CONCAT(p.path_ids, [e.entity_id]) as path_ids,
                    p.depth + 1 as depth
                FROM path p
                JOIN Relationship r ON p.entity_id = r.source_entity_id
                JOIN Entity e ON r.target_entity_id = e.entity_id
                WHERE p.depth < {max_depth}
                AND e.entity_id NOT IN UNNEST(p.path_ids)
            )
            SELECT * FROM path
            WHERE entity_id = @target_id
            """
            paths = self.execute_gql(query, {
                "source_id": source_id,
                "target_id": target_id
            })
            logger.info(f"Found {len(paths)} paths")
            return paths
        except Exception as e:
            logger.error(f"Path finding failed: {e}")
            return []
