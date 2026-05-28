from .local_files import LocalFileConnector

def get_connector(config, spark=None):
    connector_type = config["data"]["connector"]

    if connector_type == "local_files":
        return LocalFileConnector(config)

    if connector_type == "databricks":
        from .databricks import DatabricksConnector
        if spark is None:
            raise ValueError("Databricks connector requires a Spark session.")
        return DatabricksConnector(config, spark)

    raise ValueError(f"Unknown connector type: {connector_type}")