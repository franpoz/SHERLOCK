from sherlockpipe.ml.ml_model_builder import MLModelBuilder

cpus = 1
first_negative_sector = 1
#ml_training_set_preparer = MlTrainingSetPreparer("training_data/", "/home/martin/")
#ml_training_set_preparer.prepare_positive_training_dataset(cpus)
# #ml_training_set_preparer.prepare_false_positive_training_dataset(cpus)
# ml_training_set_preparer.prepare_negative_training_dataset(first_negative_sector, cpus)
#MLSingleTransitsClassifier().load_candidate_single_transits("/mnt/DATA-2/training_data/", "tp")
MLModelBuilder().get_model()
#MLModelBuilder()get_single_transit_model()
#TODO prepare_negative_training_dataset(negative_dir)
