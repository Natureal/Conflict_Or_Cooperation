### Paper: Predicting Social Unrest Using GDELT - Galla. 2018

Keywords: Social unrest, News media, GDELT, Themes, Events, Random forest, Ada boost with random forest, LSTM, County level USA

#### Hypothesis:

1. News reflects society and can be used to detect building unrest.

2. A region that is subject to higher occurrences of unrest events might suffer from a buildup of higher levels of social unrest.

3.

#### Data:

1. GDELT event table

  Used fields: (1) SQLDATE (2) EventRootCode (3) NumMentions (4) AvgTone (5) Location (Maybe is ActionGeo_**)

  Used eventRootCode: (1) 110 Disapprove (2) 130 Threaten (3) 170 Coerce (4) 145 Protest violently, riot (5) 182 Physically assault

  **Comment: 120 reject should be considered**

2. GKG Global Knowledge Graph (contains 27 fields)

  Used fields: (1) Location (2) Date (3) Themes (4) Sentiment

3.


#### Objects:

1. Using news media to predict social unrest at country and state levels for USA. (2918 counties)

2.


---
