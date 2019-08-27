# Docs

The folder for parsed documents. The format is tsv;  
Each line contains three info: (line number, sentence body, credibility score(average)).
If the latest item is `0.0`, it means there is no annotation for the sentence.
It is recommended to smartly exclude those unannotated sentences for your models.
