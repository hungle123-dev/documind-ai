def configure_system_trust_store() -> None:
    """Use the operating system certificate store for outbound HTTPS clients."""
    import truststore

    truststore.inject_into_ssl()
