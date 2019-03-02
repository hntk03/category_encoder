# category_encoder
Count Encoding and Label Count Encoding

# How to use

```python

from category_encoder import CategoryEncoder

ce = CategoryEncoder(train_df, test_df, 'Id', 'Target')
ce.encoding()
train_df, test_df, feats = ce.get_df()

```
