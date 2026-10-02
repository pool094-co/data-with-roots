import random
import numpy as np
from sklearn.linear_model import SGDRegressor

# --- 1. ENVIRONMENT DEFINITION (10x10) --- \ --- 1. DEFINICIÓN DEL ENTORNO (10x10) ---
GRID = [
    [0, 0, 0, 1, 0, 0, 2, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 2, 0, 0, 1, 0],
    [0, 1, 0, 2, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 0, 2, 0],
    [2, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 2, 0, 1, 1, 0, 0],
    [0, 2, 0, 0, 0, 0, 0, 0, 0, 2],
    [0, 0, 0, 1, 1, 0, 2, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 2, 0]
]

START = (0, 0)
GOAL = (9, 9)

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ["Up", "Down", "Left", "Right"]

ROWS = len(GRID)
COLUMNS = len(GRID[0])
NUMBER_OF_ACTIONS = len(ACTIONS)
NUMBER_OF_FEATURES = ROWS * COLUMNS * NUMBER_OF_ACTIONS

# --- 2. INTERACTION AND PREDICTION FUNCTIONS --- \ --- 2. FUNCIONES DE INTERACCIÓN Y PREDICCIÓN ---

def step(state, action):
    r"""Returns the next state, the reward, and the termination flag."""
    row = state[0] + ACTIONS[action][0]
    column = state[1] + ACTIONS[action][1]
    
    if not (0 <= row < ROWS and 0 <= column < COLUMNS) or GRID[row][column] == 1:
        return state, -10, False
        
    next_state = (row, column)
    
    if next_state == GOAL:
        return next_state, 100, True
        
    if GRID[row][column] == 2:
        return next_state, -50, False
        
    return next_state, -1, False

def encode(state, action):
    r"""Encodes a state-action pair as a one-hot vector (400 features)."""
    features = np.zeros(NUMBER_OF_FEATURES, dtype=float)
    state_index = state[0] * COLUMNS + state[1]
    feature_index = state_index * NUMBER_OF_ACTIONS + action
    features[feature_index] = 1.0
    return features

def predict_q_values(model, state):
    r"""Lectura rápida de predicciones para evitar la sobrecarga de model.predict en Render."""
    state_idx = state[0] * COLUMNS + state[1]
    base_idx = state_idx * NUMBER_OF_ACTIONS
    return model.coef_[base_idx : base_idx + NUMBER_OF_ACTIONS]

# --- 3. TRAINING CYCLE (SGDRegressor with Batching for Render) --- \ --- 3. CICLO DE ENTRENAMIENTO ---

def train(episodes=1000):
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    
    rng = random.Random(42)
    gamma = 0.95
    epsilon = 1.0
    
    # Modelo EXACTO como lo pide la rúbrica
    model = SGDRegressor(
        loss="squared_error",
        penalty=None,
        fit_intercept=False,
        learning_rate="constant",
        eta0=0.1,
        random_state=42
    )
    
    # Inicialización requerida para crear los pesos (model.coef_)
    model.partial_fit(np.zeros((1, NUMBER_OF_FEATURES)), np.array([0.0]))
    
    successes = 0
    rewards = []
    
    # FASE DE ENTRENAMIENTO
    for _ in range(episodes):
        state = START
        total_reward = 0
        
        # Listas para el Entrenamiento por Lotes (Batching)
        X_batch = []
        y_batch = []
        
        for _ in range(120):
            q_vals = predict_q_values(model, state)
            
            # Epsilon-Greedy
            if rng.random() < epsilon:
                action = rng.randrange(NUMBER_OF_ACTIONS)
            else:
                max_q = q_vals.max()
                best_actions = np.flatnonzero(q_vals == max_q).tolist()
                action = rng.choice(best_actions)
                
            next_state, reward, terminated = step(state, action)
            
            # Cálculo del valor objetivo
            if terminated:
                target = float(reward)
            else:
                next_q_vals = predict_q_values(model, next_state)
                target = reward + gamma * float(next_q_vals.max())
                
            # Guardamos la experiencia en el lote en lugar de entrenar paso a paso
            X_batch.append(encode(state, action))
            y_batch.append(target)
            
            state = next_state
            total_reward += reward
            
            if terminated:
                successes += 1
                break
                
        # ¡Magia para Render! Entrenamos el SGDRegressor 1 sola vez por episodio usando partial_fit
        model.partial_fit(X_batch, y_batch)
        
        rewards.append(total_reward)
        epsilon = max(0.05, epsilon * 0.995)
        
    # FASE DE EVALUACIÓN
    state = START
    path = [state]
    steps = []
    
    for number in range(1, 101):
        q_values = predict_q_values(model, state)
        action = int(np.argmax(q_values))
        next_state, reward, terminated = step(state, action)
        
        steps.append({
            "number": number, "state": state, "action": ACTION_NAMES[action],
            "next_state": next_state, "reward": reward, 
            "cell_type": GRID[next_state[0]][next_state[1]] if next_state != GOAL else "Goal"
        })
        
        path.append(next_state)
        state = next_state
        if terminated:
            break
            
    reached_goal = state == GOAL
    
    # CONSTRUCCIÓN DE LA TABLA Q PARA LA VISTA WEB
    q_table = []
    for row in range(ROWS):
        for column in range(COLUMNS):
            position = (row, column)
            if GRID[row][column] == 0 and position != GOAL:
                q_table.append({
                    "state": position,
                    "action_values": predict_q_values(model, position).tolist()
                })
                
    return {
        "episodes": episodes,
        "successes": successes,
        "final_average": round(sum(rewards[-100:]) / len(rewards[-100:]), 2),
        "final_epsilon": round(epsilon, 4),
        "reached_goal": reached_goal,
        "path": path,
        "steps": steps,
        "q_table": q_table,
        "eval_total_reward": sum(s["reward"] for s in steps)
    }