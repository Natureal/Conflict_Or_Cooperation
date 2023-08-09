PDF: https://dl.acm.org/doi/pdf/10.1145/3341105.3373929

Access: https://dl.acm.org/doi/pdf/10.1145/3341105.3373929

GDELT dataset: https://www.gdeltproject.org/ (It is recommended to access it through Google BigQuery)

Abstract:

International relations analysis is crucial to many stakeholders including policy makers, executives in international companies or social scientists. Generally, recent events between two countries define the international relations between them. We explore the possibilities of predicting future tendency of international relations by analyzing historical events between countries. Using auto-coded event database GDELT (Global Data on Events, Location, and Tone), which records what happened between various countries in the past few decades, we extract various types of events between two countries of interest and aggregate them into categories: conflict and cooperation. Then, according to a sequence of recent events, we predict the number of conflict events and cooperation events in the next time unit. We use MILSTM (Multi-input LSTM) considering diverse kinds of relations between different country pairs. We assume that relations between a specific pair of countries could be affected by other related country pairs. Based on this hypothesis we first select country pairs related to the target pair, and extract their multiple historical event sequences as additional input to train the model. The test results show that MILSTM performs better than vanilla LSTM, which confirms our initial hypothesis.



Citation:

Peng Chen, Adam Jatowt, and Masatoshi Yoshikawa. "Conflict or cooperation? predicting future tendency of international relations." In Proceedings of the 35th Annual ACM Symposium on Applied Computing, pp. 923-930. 2020.

@inproceedings{chen2020conflict,
  title={Conflict or cooperation? predicting future tendency of international relations},
  author={Chen, Peng and Jatowt, Adam and Yoshikawa, Masatoshi},
  booktitle={Proceedings of the 35th Annual ACM Symposium on Applied Computing},
  pages={923--930},
  year={2020}
}
