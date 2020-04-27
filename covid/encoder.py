from sklearn.preprocessing import LabelEncoder

class Encoder:

  def __init__(self):
    self.encoding_dict = {}

  def encode_dataframe(self, df):
    cols_list = []
    cols_list = df.select_dtypes(include=['object']).columns

    for col in cols_list:
      le = LabelEncoder()
      if col in self.encoding_dict:
        le = self.encoding_dict[col]
      else:
        le.fit(df[col])
    
      df[col] = le.transform(df[col])
      self.encoding_dict[col] = le
    return df

  def decode_dataframe(self, df):
    for col in self.encoding_dict:
      try:
        df[col] = self.encoding_dict[col].inverse_transform(df[col])
      except KeyError:
        pass
    return df

  def decode_column(self, column, values):
    if column in self.encoding_dict:
      return self.encoding_dict[column].inverse_transform(values)
    return values