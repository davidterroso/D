from pandas import read_csv

def test_model (
        fold_test,
        model_name,
    ):
    """
    Function used to test the trained models

    Args:
        fold_test (int): number of the fold that will be used 
            in the network testing 
        model_name (str): name of the model that will be 
            evaluated 
    """
    df = read_csv("splits/segmentation_test_splits.csv")
    test_fold_column_name = f"Fold{fold_test}_Volumes"
    test_volumes = df[test_fold_column_name].dropna().to_list()
    
if __name__ == "__main__":
    test_model(
        fold_test=2,
        model_name="Run1_UNet_best_model.pth"
    )