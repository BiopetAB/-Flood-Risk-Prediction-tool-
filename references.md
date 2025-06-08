# References

This document contains a list of references used in the development of this project

## Websites
https://ideal-postcodes.co.uk/guides/uk-postcode-format

https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/283357/ILRSpecification2013_14Appendix_C_Dec2012_v1.pdf 

https://webarchive.nationalarchives.gov.uk/20081023180830/http://www.ordnancesurvey.co.uk/oswebsite/gps/information/coordinatesystemsinfo/guidecontents/index.html 

https://medium.com/@pether.maciejewski/python-with-pgeocode-and-pandas-postal-codes-to-geo-coordinates-f75af689cc56 

https://www.geopostcodes.com/blog/python-zip-code-map/ 

https://naturaldisasters.ai/posts/python-geopandas-world-map-tutorial/#:~:text=It%20offers%20an%20in%2Dbuilt,produce%20a%20basic%20world%20map.

https://geopandas.org/docs/user_guide/mapping.html

https://www.naturalearthdata.com/downloads/50m-cultural-vectors/

https://stackoverflow.com/questions/52281173/how-to-best-use-zipcodes-in-random-forest-model-training

https://stackoverflow.com/questions/41045548/merging-changes-from-master-into-my-branch

https://blog.csdn.net/2301_81125272/article/details/139096273?ops_request_misc=%257B%2522request%255Fid%2522%253A%25223b470e4123b70ca032c3d17787fd2fb9%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=3b470e4123b70ca032c3d17787fd2fb9&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-1-139096273-null-null.142^v100^pc_search_result_base5&utm_term=.map%28%29%E5%9C%A8pandas%E4%B8%AD%E7%9A%84%E5%BA%94%E7%94%A8&spm=1018.2226.3001.4187


Books
Journal Articles
GPT
How to split postcode into two parts by space?
Answer: Use the split(" ") method to split postcode into two parts.

(e.g.): for postcode in postcodes: parts = postcode.split(" ")




## Books
NA

## Journal Articles
NA

## AI usage
Github Copilot was used throughout this project to assist in autocompleting documentation and completing repeating functions/code.

* https://chatgpt.com/share/673f31c0-cc08-8002-b4fd-bf1ee578d773 [Daniel]
    * The above ChatGPT session was used to help with the following:
        1. Visualization & interacting with the shape files
        2. Git commands, how to swich branches and merge
        3. Generating test values and troubleshooting how to verify UK post codes
        4. Developing the functions dealing with easting_northing and get_gps information
        5. How to run flake8 commands
        6. Validating code

* https://chatgpt.com/share/673fe1d5-5338-8002-93b4-60d9b1de8707 [Daniel]
    * The above ChatGPT session was used to help with the following:
        1. Merging pandas dataframes
        2. String operations
        3. Troubleshooting merge items
        4. Setup files

* https://chatgpt.com/share/673fe260-44b4-8002-98a2-51f7047eb357 [Daniel]
    * The above ChatGPT session was used to help with the following:
        1. Pandas dataframe filtering, merging, and operational syntax
        2. Git support
        3. Licensing recommendations
        4. General debugging.

* https://chatgpt.com/share/67404cfc-1798-800b-94f1-7724ed2285c4 [Si]
    * The above ChatGPT session was used to help with Code Explanation, Model Fitting, Error Debugging and Evaluation Metrics

* https://chatgpt.com/share/673fce75-9180-800b-8cbf-b97f0eb94f92 [Xianting]
    * When debugging a model training method, I encountered the error UnboundLocalError: cannot access local variable 'X_tran'. 
 
* https://chatgpt.com/share/673fceac-a9d8-800b-a449-7cba910d7e36 [Xianting]
    * When debugging a model training method, a problem occurred in the pipeline. The above ChatGPT session assisted in fixing it 
 
* https://chatgpt.com/share/673fcec8-b0dc-800b-a3ca-238f46be36c7 [Xianting]
    * The above ChatGPT session was used to show how to use pipeline to automatically get column names.
 
*  https://chatgpt.com/share/673fcf40-89e8-800b-85be-7f7590a77479 [Xianting]
    * The above ChatGPT session was used for Code Debugging and Suggestions to resolve the AttributeError error caused by the custom log_transform step not supporting the get_feature_names_out() method when using ColumnTransformer and Pipeline.

*  https://chatgpt.com/share/673fcf98-10a0-800b-83c3-29e121fc1c07 [Xianting]
    * The above ChatGPT session assisted in fixing the TypeError: list indices must be integers or slices, not str: 
 
* https://chatgpt.com/share/673fcfbe-bfd8-800b-a108-ae1de6b21441 [Xianting]
    * The above ChatGPT session assisted in understanding how to deal with postcode.
 
* https://chatgpt.com/share/673fd008-c660-800b-a836-80a0345c1f6c [Xianting]
    * The above ChatGPT session assisted in debugging errors
 
* Chat GPT prompt: [Kailun]
   * How to split postcode into two parts by space?
   *    Answer: Use the split(" ") method to split postcode into two parts.
   *    (e.g.): for postcode in postcodes: parts = postcode.split(" ")

* https://chatgpt.com/share/67405711-c754-8003-ae24-21d9da18267b [Zhihan]
   * The above ChatGPT session assisted debugging the KNN classifier due to type mismatches
