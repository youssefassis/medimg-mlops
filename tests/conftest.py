import os
import tempfile

# Keep Airflow from creating ~/airflow when the DAG tests import it.
os.environ.setdefault("AIRFLOW_HOME", tempfile.mkdtemp(prefix="airflow-home-"))
