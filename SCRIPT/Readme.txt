--------------------------------------------
------ 		SCRIPT
---------------------------------------------

Comtains the following script

1.	ALL_MAIN: this script runs the classify class on all the features and labels
	before going ahead to make any prediction

2.	DATA_PREPROCESS_MAIN: Script responsible for processing the dmoz dataset.
	after extracting token of eah uri and dropping the prefixes and suffixes, 
	it then convert the categorial variables to numerical vectors.

3.	MAIN: while ALL_MAIN caters for all the dataset, some computer may not have the requirements 
	to process big data, how much more creating converting the 1+Million dataset from categorical 
	to numerical dataset. Hence, this script takes care of this. depending on the amount of data 
	we would like to work with, MAIN can be configured to cut the data into small chunck.

4.	Playground: Script containing all sample codes used during the developing of the program.

5.	PREDICTIVE_MODEL: Script containing the classify class. the class further contains the support
	vector classifier model, random forest and Recurrent network modules.
	Each models implements a 10-fold cross-validation showing the result of each 
	folds classification report, confusion matrix and 
	precision_recall_fscore_support.

-----------------------------------------------------------------------
Others
--------------------------------
feature.pkl, label.pkl, etc are al processed pickel files of the tokenized URI.
Hence, there will not be need to run the preprocess script as these are the results.