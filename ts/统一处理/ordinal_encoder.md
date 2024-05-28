from sklearn.preprocessing import OrdinalEncoder

data = {'Category': ['A', 'B', 'A', 'C', 'B', 'C', 'A']}
df = pd.DataFrame(data)

encoder = OrdinalEncoder()

encoded_data = encoder.fit_transform(df[['Category']])

df['Encoded'] = encoded_data

category_dict = {category: int(code) for category, code in zip(encoder.categories_[0], range(len(encoder.categories_[0])))}
