import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
train_data = pd.read_csv('C:\\Users\\DELL\\Desktop\\Dataset\\train.csv')

# Display the first few rows and the basic info
print(train_data.head())
print(train_data.info())
print(train_data.describe())
# Check for missing values
missing_values = train_data.isnull().sum()
print(missing_values)

# Handle missing values
# Example: Fill missing 'Age' with the median age
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
# Fill missing 'Embarked' with the most frequent value
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)
# Drop 'Cabin' as it has too many missing values
train_data.drop(columns=['Cabin'], inplace=True)

# Create a new feature 'FamilySize'
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
# Create a new feature 'IsAlone'
train_data['IsAlone'] = 1

# Convert 'Sex' to numerical values
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
# Convert 'Embarked' to numerical values
train_data['Embarked'] = train_data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Plot the distribution of age
sns.histplot(train_data['Age'], kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Survival rate by gender
sns.barplot(x='Sex', y='Survived', data=train_data)
plt.title('Survival Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Survival Rate')
plt.xticks(ticks=[0, 1], labels=['Male', 'Female'])
plt.show()
# Survival rate by class
sns.barplot(x='Pclass', y='Survived', data=train_data)
plt.title('Survival Rate by Pclass')
plt.xlabel('Pclass')
plt.ylabel('Survival Rate')
plt.show()
# Survival rate by embarked
sns.barplot(x='Embarked', y='Survived', data=train_data)
plt.title('Survival Rate by Embarked')
plt.xlabel('Embarked')
plt.ylabel('Survival Rate')
plt.xticks(ticks=[0, 1, 2], labels=['C', 'Q', 'S'])
plt.show()
# Survival rate by family size
sns.barplot(x='FamilySize', y='Survived', data=train_data)
plt.title('Survival Rate by Family Size')
plt.xlabel('Family Size')
plt.ylabel('Survival Rate')
plt.show()

# Compute the correlation matrix
train_data.drop(columns=['Name'], inplace=True)
train_data.drop(columns=['Ticket'], inplace=True)
corr = train_data.corr()
# Generate a heatmap
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
