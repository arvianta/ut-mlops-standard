import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
import os
import yaml

from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime

seed = 42
tf.random.set_seed(seed)

# Step 1: Custom Balanced Accuracy Metric (unchanged)
class BalancedAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='balanced_accuracy', **kwargs):
        super(BalancedAccuracy, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.true_negatives = self.add_weight(name='tn', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        tp = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32))
        tn = tf.reduce_sum(tf.cast((1 - y_true) * (1 - y_pred), tf.float32))
        fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, tf.float32))
        fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), tf.float32))

        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.true_negatives.assign_add(tn)
        self.false_negatives.assign_add(fn)
    
    def result(self):
        recall_pos = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
        recall_neg = self.true_negatives / (self.true_negatives + self.false_positives + tf.keras.backend.epsilon())
        
        recall_pos = tf.clip_by_value(recall_pos, 0, 1)
        recall_neg = tf.clip_by_value(recall_neg, 0, 1)
        
        balanced_acc = (recall_pos + recall_neg) / 2
        return balanced_acc

    def reset_states(self):
        self.true_positives.assign(0.0)
        self.false_positives.assign(0.0)
        self.true_negatives.assign(0.0)
        self.false_negatives.assign(0.0)

def build_model(hp):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.BatchNormalization(input_shape=(43,)))
    model.add(tf.keras.layers.Dropout(hp.Choice('dropout_1', values=[0.2, 0.3, 0.4])))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(hp.Choice('dropout_2', values=[0.2, 0.3, 0.4])))
    model.add(tf.keras.layers.Dense(128, activation='relu'))

    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(hp.Choice('dropout_3', values=[0.2, 0.3, 0.4])))
    model.add(tf.keras.layers.Dense(64, activation='relu'))

    model.add(tf.keras.layers.Dense(3, activation='softmax'))

    learning_rate = hp.Choice('learning_rate', values=[1e-4, 1e-3, 1e-2])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy', BalancedAccuracy()])
    
    return model

class TuningMLflowCallback(tf.keras.callbacks.Callback):
    def __init__(self, trial_id, hyperparameters):
        super().__init__()
        self.trial_id = trial_id
        self.hyperparameters = hyperparameters
        
    def on_train_begin(self, logs=None):
        model_name = f"Trial_{self.trial_id}_d{self.hyperparameters.get('dropout_1')}_{self.hyperparameters.get('dropout_2')}_{self.hyperparameters.get('dropout_3')}_lr{self.hyperparameters.get('learning_rate')}"
        self.run = mlflow.start_run(run_name=model_name, nested=True)
        mlflow.log_params(self.hyperparameters)
        
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            mlflow.log_metrics({
                "balanced_accuracy": logs.get('balanced_accuracy', 0),
                "accuracy": logs.get('accuracy', 0),
                "loss": logs.get('loss', 0),
                "epoch": epoch
            }, step=epoch)
    
    def on_train_end(self, logs=None):
        mlflow.end_run()


def evaluate_model(model, X_test, Y_test, run_id=None):
    """
    Evaluate the model and log metrics to MLflow
    """
    # Get predictions
    Y_pred = model.predict(X_test)
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    Y_test_classes = np.argmax(Y_test, axis=1)
    
    # Calculate metrics
    class_report = classification_report(Y_test_classes, Y_pred_classes, output_dict=True)
    conf_matrix = confusion_matrix(Y_test_classes, Y_pred_classes)
    
    # Create confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save confusion matrix plot
    confusion_matrix_path = 'confusion_matrix.png'
    plt.savefig(confusion_matrix_path)
    plt.close()
    
    # Calculate additional metrics
    metrics = {
        'test_accuracy': class_report['accuracy'],
        'test_macro_avg_precision': class_report['macro avg']['precision'],
        'test_macro_avg_recall': class_report['macro avg']['recall'],
        'test_macro_avg_f1': class_report['macro avg']['f1-score'],
        'test_weighted_avg_precision': class_report['weighted avg']['precision'],
        'test_weighted_avg_recall': class_report['weighted avg']['recall'],
        'test_weighted_avg_f1': class_report['weighted avg']['f1-score']
    }
    
    # Add per-class metrics
    for i in range(3):  # Assuming 3 classes
        metrics.update({
            f'test_class_{i}_precision': class_report[str(i)]['precision'],
            f'test_class_{i}_recall': class_report[str(i)]['recall'],
            f'test_class_{i}_f1': class_report[str(i)]['f1-score']
        })
    
    return metrics, confusion_matrix_path


def save_model_config(hyperparameters, metrics, model_dir='model_artifacts'):
    """
    Save model configuration and metrics to YAML file
    """
    os.makedirs(model_dir, exist_ok=True)
    
    config = {
        'hyperparameters': hyperparameters,
        'metrics': metrics,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    config_path = os.path.join(model_dir, 'model_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config_path