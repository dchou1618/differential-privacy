import numpy as np
import jax.numpy as jnp 
import pandas as pd

from jax import grad, jit, random
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# from paper: 

# Step 1: Data Preparation
def preprocess_data(sample_size):
    
    X, y = make_classification(n_samples=sample_size, n_features=10, n_informative=5, random_state=42)
    X = (X - X.min()) / (X.max() - X.min())  # Normalize features to [0, 1]
    return X, y

# Step 2: Client-Side Implementation (DP-SGD)
def logistic_loss(weights, X, y):
    logits = jnp.dot(X, weights)
    loss = jnp.mean(jnp.log(1 + jnp.exp(-y * logits)))
    return loss

@jit
def dp_sgd_update(weights, X, y, learning_rate, noise_scale, rng_key):
    gradient = grad(logistic_loss)(weights, X, y)
    noise = random.laplace(rng_key, shape=gradient.shape) * noise_scale
    dp_gradient = gradient + noise
    updated_weights = weights - learning_rate * dp_gradient
    return updated_weights


def illustrate_theorem1(num_models=5, model_dim=10, noise_variance=1.0, steps=100):
    """Illustrate that the variance of the weights remains k * sigma^2 / 2."""
    np.random.seed(42)
    models = [np.random.normal(0, np.sqrt(num_models * noise_variance / 2), size=model_dim) 
              for _ in range(num_models)]
    
    variance_history = []
    for _ in range(steps):
        # Randomly select a model
        selected_idx = np.random.randint(0, num_models)
        noise = np.random.normal(0, np.sqrt(noise_variance), size=model_dim)
        updated_model = models[selected_idx] + noise
        
        # Replace a random model
        replace_idx = np.random.randint(0, num_models)
        models[replace_idx] = updated_model
        
        # Calculate variance across all models
        all_models = np.array(models)
        variance = np.var(all_models, axis=0)
        variance_history.append(np.mean(variance))
    
    # Plot variance over time
    plt.plot(variance_history, label="Variance")
    plt.axhline(y=num_models * noise_variance / 2, color='r', linestyle='--', label=f"Expected Variance = {num_models * noise_variance / 2}")
    plt.title("Illustration of Theorem 1: Variance Stabilization")
    plt.xlabel("Steps")
    plt.ylabel("Variance")
    plt.legend()
    plt.show()

# Step 3: Server-Side "Draw and Discard"
class DrawAndDiscardServer:
    def __init__(self, num_models, model_dim, noise_scale):
        self.num_models = num_models
        self.models = [np.random.normal(size=model_dim) for _ in range(num_models)]
        self.noise_scale = noise_scale

    def get_model(self):
        # Randomly select a model instance
        return self.models[np.random.choice(len(self.models),size=1)[0]]

    def update_model(self, updated_model):
        # Randomly replace one of the k models
        replace_idx = np.random.randint(0, self.num_models)
        self.models[replace_idx] = updated_model

    def average_model(self):
        # Average all model instances
        return np.mean(self.models, axis=0)

for k in range(1, 5):
    for sample_size in range(100, 500000, 100):
        X, y = preprocess_data(sample_size)
        data = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        data['target'] = y

        # Initialize server
        rng_key = random.PRNGKey(42)
        server = DrawAndDiscardServer(num_models=5, model_dim=X.shape[1], noise_scale=0.1)

        # Step 4: Privacy and Performance Evaluation
        num_clients = 10
        client_data = np.array_split(data, num_clients)
        weights = np.random.normal(size=X.shape[1])
        learning_rate = 0.001

        for epoch in range(20):
            for client_idx, client_df in enumerate(client_data):
                client_X = client_df.iloc[:, :-1].values
                client_y = client_df.iloc[:, -1].values
                client_model = server.get_model()
                
                # Update the model with DP-SGD
                weights = dp_sgd_update(jnp.array(client_model), jnp.array(client_X), jnp.array(client_y), 
                                        learning_rate, noise_scale=0.1, rng_key=rng_key)
                
                server.update_model(weights)
            
            # Log progress
            avg_model = server.average_model()
            loss = logistic_loss(avg_model, jnp.array(X), jnp.array(y))
            print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

        # Step 5: Visualization
        avg_model = server.average_model()
        plt.plot(avg_model, label="Averaged Weights")
        plt.title("Averaged Model Weights")
        plt.xlabel("Feature Index")
        plt.ylabel("Weight Value")
        plt.legend()
        plt.show()

        illustrate_theorem1()