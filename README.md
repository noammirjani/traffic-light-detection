## CityScape Traffic Light Detection and Verification

### Data

For details about cityscape, see https://www.cityscapes-dataset.com/dataset-overview/
To download after login: https://www.cityscapes-dataset.com/downloads/
Need to download:
- gtFine_trainvaltest.zip (241MB)
- leftImg8bit_trainvaltest.zip (11GB)  
(Can also download the leftImg8bit_trainextra.zip (44GB), but it's pretty big...)

### Local arrangement of the files
No matter how you arrange your files, just have a CSV file with at least the following columns:  
imag_path gtim_path json_path train_test_val  
Add what you want in the comments (or ignore)  
All paths are absolute relative to the CSV folder.  

- You may try to use `data_utils.make_files_lists` to arrange the data