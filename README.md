# FreshDash 

## **Using Yelp reviews to optimize restaurant inspection** 




## Data Sources:
- [Seattle Yelp Data](http://cs.stonybrook.edu/~junkang/hygiene)
- [Seattle Food Establishment Inspection Data](https://data.kingcounty.gov/Health-Wellness/Food-Establishment-Inspection-Data/f29f-zza5)

## Code Structure

**/Data**
- /features: features tables.
- /Seattle: put raw Yelp review, merged inspection instances, and inspection record data here.

**/Data Exploration**
- Link inspection record
- Data cleaning
- Exploratory analysis and visualization

**/Feature Generation**

* /Ngram: code and output

* /Topic Modeling: 

  * Code 

  * Visualization demo

  * mallet: To replicate labeled-LDA result, 1. use R code to generate rev_tm_violation_rating.txt. 2. run command below in the terminal:

  * ```
    bin/mallet import-file --input rev_tm_violation_rating.txt --output yelp-short.seq --stoplist-file yelp.stops --label-as-features --keep-sequence --line-regex '([^\t]+)\t([^\t]+)\t(.*)'
    
    bin/mallet run cc.mallet.topics.LabeledLDA --input yelp-short.seq --output-topic-keys yelp-llda.keys --output-doc-topics docsAsTopicsProbs.txt
    ```

    

**/Machine Learning Pipeline**

* /config: machine learning model training configuration yaml files

* /mloutput: model evaluation output

* classifiers.py: model training

  * ```
    python3 classifiers.py <config file path>
    ```

* MagicLoop.ipynb: run classifiers.py in Jupyter Notebook for handy visualization and model comparison

## Contributors

Ran Bi,
Shambhavi Mohan,
Minjia Zhu
@Uchicago Harris, CAPP