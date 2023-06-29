import tensorflow as tf


new_model = tf.keras.models.load_model("model/my_model")

# Check its architecture
print(new_model.summary())

print(new_model.encoder.summary())

print(new_model.decoder.summary())

pred = new_model.predict(normal_test_data)


# Calculate train loss
reconstructions = autoencoder.predict(normal_train_data)
train_loss = tf.keras.losses.mae(reconstructions, normal_train_data)

# Get predictions
threshold = np.mean(train_loss) + np.std(train_loss)
preds = predict(autoencoder, test_data, threshold)

# Plot confusion matrix
plot_confusion_matrix(test_labels, preds)

# concat normal_test_data and pred and build dataframe
normal_test_data = pd.DataFrame(normal_test_data)
pred = pd.DataFrame(pred)
normal_test_data = pd.concat([normal_test_data, pred], axis=1)

# another column with true if both are True or both False
normal_test_data["true"] = np.where(
    (normal_test_data[0] == normal_test_data[1]), True, False
)
