# -*- coding: utf-8 -*-

"""
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler,LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

kickstarter_data = pd.read_excel(r"C:\Users\Aasna\Downloads\Kickstarter (2).xlsx")

####Random Forest#########

# 1. Pre-process Data

# Filtering the dataset for projects that are either "successful" or "failed"
filtered_kickstarter_data = kickstarter_data[kickstarter_data['state'].isin(['successful', 'failed'])]

# Select Relevant Columns
relevant_columns = [
    'goal', 'disable_communication', 'country', 'currency', 'category',
    'static_usd_rate', 'created_at_weekday', 'launched_at_weekday', 'created_at_month', 'created_at_day',
    'created_at_yr', 'created_at_hr', 'launched_at_month', 'launched_at_day',
    'launched_at_yr', 'launched_at_hr', 'create_to_launch_days', 'deadline_weekday',
    'deadline_month', 'deadline_day', 'deadline_yr', 'deadline_hr'
]

data_selected = filtered_kickstarter_data[relevant_columns].copy()

# Handle Missing Values
imputer = SimpleImputer(strategy='most_frequent')
imputed_values = imputer.fit_transform(data_selected[['category']])
data_selected.loc[:, ['category']] = imputed_values

# One-hot encoding the weekday columns
weekday_columns = ['created_at_weekday', 'launched_at_weekday', 'deadline_weekday']
encoder = OneHotEncoder(sparse=False)
encoded_weekdays = encoder.fit_transform(data_selected[weekday_columns])
encoded_weekdays_df = pd.DataFrame(encoded_weekdays, columns=encoder.get_feature_names_out(weekday_columns))
data_encoded_train= data_selected.drop(weekday_columns, axis=1)

# Encode Categorical Variables
categorical_columns = ['country', 'currency', 'category']
one_hot_encoder = OneHotEncoder(sparse=False)
encoded_categorical = one_hot_encoder.fit_transform(data_encoded_train[categorical_columns])
encoded_categorical_df = pd.DataFrame(encoded_categorical, columns=one_hot_encoder.get_feature_names_out(categorical_columns))

# Reset indices to align rows correctly
encoded_weekdays_df.reset_index(drop=True, inplace=True)
encoded_categorical_df.reset_index(drop=True, inplace=True)
data_encoded_train.reset_index(drop=True, inplace=True)

data_encoded_train = pd.concat([data_encoded_train.drop(categorical_columns, axis=1), encoded_weekdays_df, encoded_categorical_df], axis=1)

# Scale Numerical Variables
scaler = StandardScaler()
data_encoded_train[['goal', 'static_usd_rate']] = scaler.fit_transform(data_encoded_train[['goal', 'static_usd_rate']])

# Encoding the target variable
target = filtered_kickstarter_data['state'].replace({'successful': 1, 'failed': 0})

# Check if the lengths match
assert len(data_encoded_train) == len(target)


#print(data_encoded.head())

#####First Random Forest######

# Splitting the data into features and target variable
X = data_encoded_train
y = target 

# Splitting the data into features and target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Developing a Random Forest model
rf_classifier = RandomForestClassifier(random_state=0)
rf_classifier.fit(X_train, y_train)

# Making predictions
y_pred = rf_classifier.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Output the accuracy and classification report
print('Accuracy of the Random Forest classifier:', accuracy)
print('Classification report:\n', report)


##########First Hyperparameter Tuning##########

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=0)

# Initialize the GridSearchCV
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Perform the grid search
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
best_params

print("Best hyperparameters:", best_params)

# Train the Random Forest classifier with the best hyperparameters
rf_classifier_best = RandomForestClassifier(**best_params, random_state=0)

# Fit the model
rf_classifier_best.fit(X_train, y_train)

# Make predictions
y_pred_best = rf_classifier_best.predict(X_test)

# Evaluate the classifier
accuracy_best = accuracy_score(y_test, y_pred_best)
class_report_best = classification_report(y_test, y_pred_best)


print('Accuracy of the Random Forest classifier with best hyperparameters:', accuracy_best)
print('Classification report with best hyperparameters:\n', class_report_best)



# Train the Random Forest classifier with the best hyperparameters and class weights adjusted

# Train the Random Forest classifier with the best hyperparameters and class weights adjusted
rf_classifier_best = RandomForestClassifier(
    max_depth=30,
    min_samples_leaf=1,
    min_samples_split=10,
    n_estimators=300,
    random_state=0,
    class_weight='balanced'  # Adjust the class weights to 'balanced'
)

# Fit the model
rf_classifier_best.fit(X_train, y_train)

# Make predictions
y_pred_best = rf_classifier_best.predict(X_test)

# Evaluate the classifier
accuracy_best = accuracy_score(y_test, y_pred_best)
class_report_best = classification_report(y_test, y_pred_best)

print('Accuracy of the Random Forest classifier with best hyperparameters and balanced weights:', accuracy_best)
print('Classification report with best hyperparameters:\n', class_report_best)



##########Testing to find the optimal set of weights that performs better than balanced weight#############

# Define a range of weights to try for Class 1
class_weights_to_try = [
    {0: 1, 1: 2},  # Double the weight for Class 1
    {0: 1, 1: 3},  # Triple the weight for Class 1
    {0: 1, 1: 4},  # Quadruple the weight for Class 1
]

# Store the best score and corresponding class weight
best_score = 0
best_weights = None
best_report = None

# Loop over the different class weights to find the best
for weights in class_weights_to_try:
    # Initialize the classifier with the current class weights
    rf_classifier_optimal = RandomForestClassifier(
        max_depth=30,
        min_samples_leaf=1,
        min_samples_split=10,
        n_estimators=300,
        random_state=0,
        class_weight=weights
    )
    
    
    # Train the classifier
    rf_classifier.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_classifier.predict(X_test)
    
    # Evaluate the classifier using classification report or any other method
    report = classification_report(y_test, y_pred, output_dict=True)
    score = report['1']['f1-score']  # You can choose precision or recall based on your requirement
    
    # Update the best score and weights if current score is better
    if score > best_score:
        best_score = score
        best_weights = weights
        best_report = report

# Output the best weights and corresponding score
print(f'Best F1-Score for Class 1: {best_score}')
print(f'Best class weights: {best_weights}')
print(f'Classification report for the best weights:\n {best_report}')



#########Applying optimal weights#################c

# Retrain the Random Forest classifier with the optimal class weight
rf_classifier_best = RandomForestClassifier(
    max_depth=30,
    min_samples_leaf=1,
    min_samples_split=10,
    n_estimators=300,
    random_state=0,
    class_weight={0: 1, 1: 2}  # Optimal class weight found
)

# Fit the model on the training data
rf_classifier_optimal.fit(X_train, y_train)

# Make predictions on the test data
y_pred_optimal = rf_classifier_optimal.predict(X_test)

# Evaluate the classifier
accuracy_optimal = accuracy_score(y_test, y_pred_optimal)
class_report_optimal = classification_report(y_test, y_pred_optimal)

# Output the accuracy and classification report
print('Accuracy of the Random Forest classifier with optimal class weights:', accuracy_optimal)
print('Classification report with optimal class weights:\n', class_report_optimal)

##########FINAL RANDOM FOREST MODEL##################

#the overall accuracy with optimal weights is lower and the weight distribution is not substantial
#the final model is the hypertuned + balanced weights model  


rf_classifier_best = RandomForestClassifier(
    max_depth=30,
    min_samples_leaf=1,
    min_samples_split=10,
    n_estimators=300,
    random_state=0,
    class_weight='balanced'  # Adjust the class weights to 'balanced'
)

# Fit the model
rf_classifier_best.fit(X_train, y_train)

# Make predictions
y_pred_best = rf_classifier_best.predict(X_test)

# Evaluate the classifier
accuracy_best = accuracy_score(y_test, y_pred_best)
class_report_best = classification_report(y_test, y_pred_best)


print('Accuracy of the Random Forest classifier with best hyperparameters and balanced weights:', accuracy_best)
print('Classification report with best hyperparameters:\n', class_report_best)


#########K-MEANS CLUSTERING###############

######Pre-processing + elbow plot to find optimal clusters########

#Preprocessing

# Filtering the dataset for projects that are either "successful" or "failed"
filtered_data = kickstarter_data[kickstarter_data['state'].isin(['successful', 'failed'])].copy()

# Selecting relevant features for clustering
features = ['goal', 'pledged', 'create_to_launch_days', 'launch_to_deadline_days', 'launch_to_state_change_days']
clustering_data = filtered_data[features]

# Normalizing the data
scaler = StandardScaler()
clustering_data_normalized = scaler.fit_transform(clustering_data)

# Find the optimal number of clusters using the elbow method
wcss = []
for i in range(1, 11):  # Trying values of k from 1 to 10
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(clustering_data_normalized)
    wcss.append(kmeans.inertia_)

# Plot the WCSS values to identify the elbow point
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='-', color='b')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.grid(True)
plt.show()

###########Applying first K-Means clustering###########

# Applying k-means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(clustering_data_normalized)

# Adding the cluster labels to the original dataframe
filtered_data['cluster'] = kmeans.labels_

# Grouping the data by clusters and calculating mean values of the features
cluster_characteristics = filtered_data.groupby('cluster')[features].mean()
print("\nCluster Characteristics:\n", cluster_characteristics)

# Calculate the count of data points in each cluster
cluster_counts = filtered_data['cluster'].value_counts()

# Output the counts
print("Counts of Data Points in Each Cluster:\n", cluster_counts)

######### First PCA Plot###############

# Applying PCA to reduce the data to 2 dimensions for visualization
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(clustering_data_normalized)

# Plotting the clusters
plt.figure(figsize=(10, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmeans.labels_, cmap='viridis', marker='o', edgecolor='k', s=50)
plt.title('Clusters visualization with 2D PCA')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.colorbar(label='Cluster Label')
plt.show()


##Re-evaluating optimal number of clusters + cluster distribution based on PCA plot + only 5 in one clusters/ charactersitics not distinct enough

####Adding success rate as a new feature########

# Feature Engineering

# Calculate the funding success rate and create a new feature "success_rate"
kickstarter_data['success_rate'] = kickstarter_data['usd_pledged'] / kickstarter_data['goal']


# Select relevant features for clustering (including the newly engineered features)
features = [
    'goal', 'pledged', 'success_rate','create_to_launch_days', 'launch_to_deadline_days', 'launch_to_state_change_days'
]

 
# Normalize the data
scaler = StandardScaler()
clustering_data_normalized = scaler.fit_transform(kickstarter_data[features])

# Applying k-means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(clustering_data_normalized)

# Adding the cluster labels to the original DataFrame
kickstarter_data['cluster'] = kmeans.labels_

# Grouping the data by clusters and calculating mean values of the features
cluster_characteristics = kickstarter_data.groupby('cluster')[features].mean()

# Output cluster characteristics
print("Cluster Characteristics:\n", cluster_characteristics)

cluster_counts = kickstarter_data['cluster'].value_counts()
print(cluster_counts)

##### Second PCA######
# Applying PCA to reduce the data to 2 dimensions for visualization
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(clustering_data_normalized)

# Plotting the clusters
plt.figure(figsize=(10, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmeans.labels_, cmap='viridis', marker='o', edgecolor='k', s=50)
plt.title('Clusters visualization with 2D PCA')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.colorbar(label='Cluster Label')
plt.show()

### Add new feature - avg pledge amount per backer 

# Calculate the average pledge amount per backer and handle potential division by zero
kickstarter_data['avg_pledge_per_backer'] = kickstarter_data['pledged'] / kickstarter_data['backers_count'].replace(0, np.nan)
kickstarter_data['avg_pledge_per_backer'].fillna(0, inplace=True)  # Fill NaN values if any


kickstarter_data = pd.concat([kickstarter_data], axis=1)

# Select relevant features for clustering
numeric_features = [
    'goal', 'pledged', 'success_rate', 'create_to_launch_days', 
    'launch_to_deadline_days', 'launch_to_state_change_days', 'avg_pledge_per_backer',
]
features = numeric_features 
# Normalize the data
scaler = StandardScaler()
clustering_data_normalized = scaler.fit_transform(kickstarter_data[features])

# Applying k-means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(clustering_data_normalized)

# Adding the cluster labels to the original DataFrame
kickstarter_data['cluster'] = kmeans.labels_

# Grouping the data by clusters and calculating mean values of the features
cluster_characteristics = kickstarter_data.groupby('cluster')[numeric_features].mean()

# Output cluster characteristics
print("Cluster Characteristics:\n", cluster_characteristics)

####### PCA THREE #########
# Applying PCA to reduce the data to 2 dimensions for visualization
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(clustering_data_normalized)

# Plotting the clusters
plt.figure(figsize=(10, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmeans.labels_, cmap='viridis', marker='o', edgecolor='k', s=50)
plt.title('Clusters visualization with 2D PCA')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.colorbar(label='Cluster Label')
plt.show()

### Looking at the Cluster 3 distribution - look for outliers 

# Filter out Cluster 3 for analysis
cluster_3_data = kickstarter_data[kickstarter_data['cluster'] == 3]

# Summary statistics for Cluster 3
cluster_3_summary = cluster_3_data.describe()

# Display summary statistics to understand the distribution of Cluster 3
print("Cluster 3 Summary Statistics:")
print(cluster_3_summary)

# look at the projects with the highest pledges within Cluster 3
print("\nProjects with the highest pledges in Cluster 3:")
top_pledged_projects = cluster_3_data.sort_values(by='pledged', ascending=False).head()
print(top_pledged_projects[['name', 'goal', 'pledged', 'backers_count', 'success_rate', 'avg_pledge_per_backer']])

# check for any projects that have a very high pledge amount compared to others
print("\nPotential Outliers in Cluster 3:")
potential_outliers = cluster_3_data[cluster_3_data['pledged'] > cluster_3_data['pledged'].quantile(0.95)]
print(potential_outliers[['name', 'goal', 'pledged', 'backers_count', 'success_rate', 'avg_pledge_per_backer']])

#### remove outliers + run K-means again #####

# Identify projects in Cluster 3 with a goal of 1 unit of currency
outliers = kickstarter_data[(kickstarter_data['cluster'] == 3) & (kickstarter_data['goal'] == 1)]

# Remove these outliers from dataset
kickstarter_data_cleaned = kickstarter_data.drop(outliers.index)

features = [ 'goal', 'pledged', 'success_rate', 'create_to_launch_days', 
    'launch_to_deadline_days', 'launch_to_state_change_days', 'avg_pledge_per_backer'
]

# Normalize the data for the cleaned dataset
scaler = StandardScaler()
clustering_data_normalized_cleaned = scaler.fit_transform(kickstarter_data_cleaned[features])

# Re-run k-means clustering on the cleaned and normalized data
kmeans_cleaned = KMeans(n_clusters=3, random_state=42)
kmeans_cleaned.fit(clustering_data_normalized_cleaned)

# Adding the new cluster labels to the cleaned DataFrame
kickstarter_data_cleaned['cluster'] = kmeans_cleaned.labels_

# Now you can proceed to analyze the new cluster characteristics
cluster_characteristics_cleaned = kickstarter_data_cleaned.groupby('cluster')[features].mean()
print("New Cluster Characteristics:\n", cluster_characteristics_cleaned)

#### PCA FOUR ########
# Applying PCA to reduce the data to 2 dimensions for visualization
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(clustering_data_normalized_cleaned)

# Plotting the clusters
plt.figure(figsize=(10, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmeans_cleaned.labels_, cmap='viridis', marker='o', edgecolor='k', s=50)
plt.title('Clusters visualization with 2D PCA')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.colorbar(label='Cluster Label')
plt.show()

cluster_counts = kickstarter_data_cleaned['cluster'].value_counts()
print(cluster_counts)

##### Addressing spread of cluster 2 + overall spread of clusters #####

cluster_2_data = kickstarter_data_cleaned[kickstarter_data_cleaned['cluster'] == 2]

# Plotting histograms for all numerical features in Cluster 3
numerical_features = ['goal', 'pledged', 'backers_count', 'avg_pledge_per_backer']  # add other numerical features as needed
for feature in numerical_features:
    plt.figure(figsize=(10, 5))
    sns.histplot(cluster_2_data[feature], kde=True)
    plt.title(f'Distribution of {feature} in Cluster 3')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()


#### Apply log transformation to the skewed features to reduce skewness ######

# Log-transform the skewed features and add them to the 'kickstarter_data_cleaned' DataFrame
skewed_features = ['goal', 'pledged', 'backers_count', 'avg_pledge_per_backer']  
for feature in skewed_features:
    # Apply log transformation with a small constant to avoid issues with log(0)
    kickstarter_data_cleaned[feature + '_log'] = np.log1p(kickstarter_data_cleaned[feature] + 1)


###### Applying K-means to log transformed features + original features

# Select both the original and the new log-transformed features for clustering
features_log_transformed = [feat + '_log' for feat in skewed_features] + [
    'success_rate', 'create_to_launch_days', 'launch_to_deadline_days', 'launch_to_state_change_days'
]

# Normalize the log-transformed data
scaler = StandardScaler()
X_cleaned_log_transformed_scaled = scaler.fit_transform(kickstarter_data_cleaned[features_log_transformed])

# Re-run k-means clustering on the normalized, log-transformed data
kmeans_cleaned_log_transformed = KMeans(n_clusters=3, random_state=42)
kmeans_cleaned_log_transformed.fit(X_cleaned_log_transformed_scaled)

# Adding the new cluster labels to the cleaned DataFrame
kickstarter_data_cleaned['cluster_log_transformed'] = kmeans_cleaned_log_transformed.labels_

# Analyze the new cluster characteristics
# We use the original feature names here, but the clustering was done on the log-transformed data
cluster_characteristics_cleaned_log_transformed = kickstarter_data_cleaned.groupby('cluster_log_transformed')[features].mean()
print("New Cluster Characteristics after Log Transformation:\n", cluster_characteristics_cleaned_log_transformed)


# Calculate Silhouette Score
silhouette_avg = silhouette_score(X_cleaned_log_transformed_scaled, kmeans_cleaned_log_transformed.labels_)
print(f'Silhouette Score: {silhouette_avg}')


##Final clusters with only three clusters + log transformed features 

# Applying PCA to reduce the data to 2 dimensions for visualization
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(X_cleaned_log_transformed_scaled)

# Plotting the clusters
plt.figure(figsize=(10, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmeans_cleaned_log_transformed.labels_, cmap='viridis', marker='o', edgecolor='k', s=50)
plt.title('Clusters visualization with 2D PCA')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.colorbar(label='Cluster Label')
plt.show()

## Analyze the clusters in relation to other characteristics 

#Segmentation by Backers' Behavior
#Calculate the number of backers per project within each cluster
kickstarter_data_cleaned['backers_count_log'] = np.log1p(kickstarter_data_cleaned['backers_count'])
backers_per_cluster = kickstarter_data_cleaned.groupby('cluster_log_transformed')['backers_count_log'].mean()
print("Average Number of Backers (log-transformed) per Cluster:\n", backers_per_cluster)

#Temporal Analysis
#`creation_to_launch_days` and `launch_to_deadline_days` if not already present
temporal_features = ['create_to_launch_days', 'launch_to_deadline_days']
temporal_analysis = kickstarter_data_cleaned.groupby('cluster_log_transformed')[temporal_features].mean()
print("Temporal Features per Cluster:\n", temporal_analysis)

#Financial Metrics Examination
#Analyze the variability within clusters regarding goals and pledged amounts
financial_metrics = ['goal', 'pledged']  
financial_analysis = kickstarter_data_cleaned.groupby('cluster_log_transformed')[financial_metrics].agg(['mean', 'std'])
print("Financial Metrics per Cluster:\n", financial_analysis)

#Project Descriptors and Category Performance
#Top categories per cluster
top_categories_per_cluster = kickstarter_data_cleaned.groupby('cluster_log_transformed')['category'].agg(lambda x: x.value_counts().index[0])
print("Top Categories per Cluster:\n", top_categories_per_cluster)

#Average goal and pledged amounts per category within each cluster
category_performance = kickstarter_data_cleaned.groupby(['cluster_log_transformed', 'category'])['goal_log', 'pledged_log'].mean()
print("Category Performance (Average Goal and Pledged) per Cluster:\n", category_performance)


