from edgemodelkit import DataFetcher

# Initialize for HTTP communication
fetcher = DataFetcher(source="http", api_url="http://10.124.149.135")

# Log 10 samples with timestamp and count columns
fetcher.log_sensor_data(class_label="M0_2.3", num_samples=3000, add_timestamp=True, add_count=False)