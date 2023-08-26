from pipelines.training_pipeline import train_pipeline

from zenml.client import Client


if __name__ == '__main__':
    #run the training pipeline
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path='./data/olist_customers_dataset.csv')
    
#mlflow ui --backend-store-uri "file:/Users/prajw/AppData/Roaming/zenml/local_stores/23919a01-7610-44b9-9f51-4be666f749a3/mlruns"