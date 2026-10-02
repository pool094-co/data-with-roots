import random
import numpy as np
from sklearn.linear_model import SGDRegressor

# --- 1. ENVIRONMENT DEFINITION (10x10) --- \ --- 1. DEFINICIÓN DEL ENTORNO (10x10) ---
# Visual dictionary: 0 = Normal path (o), 1 = Wall (#), 2 = Danger Zone (D) \ Diccionario visual: 0 = Camino normal (o), 1 = Muro (#), 2 = Zona de Peligro (D)
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
    r"""Returns the next state, the reward, and the termination flag. \ Retorna el siguiente estado, la recompensa y la señal de terminación."""
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
    r"""Encodes a state-action pair as a one-hot vector. \ Codifica un par estado-acción como un vector one-hot."""
    features = np.zeros(NUMBER_OF_FEATURES, dtype=float)
    state_index = state[0] * COLUMNS + state[1]
    feature_index = state_index * NUMBER_OF_ACTIONS + action
    features[feature_index] = 1.0
    return features

def predict_q_values(model, state):
    r"""Predicts the Q-value of each available action for a given state. \ Predice el valor Q de cada acción disponible para un estado dado."""
    features = np.array([encode(state, action) for action in range(NUMBER_OF_ACTIONS)])
    return model.predict(features)


# --- 3. TRAINING CYCLE (OPTIMIZED FOR RENDER CLOUD) --- \ --- 3. CICLO DE ENTRENAMIENTO (OPTIMIZADO PARA RENDER NUBE) ---

def train(episodes=250):
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    
    rng = random.Random(42)
    gamma = 0.95
    epsilon = 1.0
    
    model = SGDRegressor(
        loss="squared_error",
        penalty=None,
        fit_intercept=False,
        learning_rate="constant",
        eta0=0.1,
        random_state=42
    )
    
    model.partial_fit(np.zeros((1, NUMBER_OF_FEATURES)), np.array([0.0]))
    
    successes = 0
    rewards = []
    
    # TRAINING PHASE \ FASE DE ENTRENAMIENTO
    for _ in range(episodes):
        state = START
        total_reward = 0
        
        for _ in range(60): # Fast step limit to avoid HTTP timeout \ Límite de pasos rápido para evitar timeout HTTP
            if rng.random() < epsilon:
                action = rng.randrange(NUMBER_OF_ACTIONS)
            else:
                q_values = predict_q_values(model, state)
                best_actions = np.flatnonzero(q_values == q_values.max()).tolist()
                action = rng.choice(best_actions)
                
            next_state, reward, terminated = step(state, action)
            
            if terminated:
                target = float(reward)
            else:
                next_q_values = predict_q_values(model, next_state)
                target = reward + gamma * float(next_q_values.max())
                
            features = encode(state, action).reshape(1, -1)
            model.partial_fit(features, np.array([target]))
            
            state = next_state
            total_reward += reward
            
            if terminated:
                successes += 1
                break
                
        rewards.append(total_reward)
        epsilon = max(0.05, epsilon * 0.98) # Decay adjusted for 250 episodes \ Decaimiento ajustado para 250 episodios
        
    # EVALUATION PHASE \ FASE DE EVALUACIÓN
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
    
    # BUILD Q-TABLE \ CONSTRUCCIÓN DE LA TABLA Q
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
        "final_average": round(sum(rewards[-50:]) / len(rewards[-50:]), 2),
        "final_epsilon": round(epsilon, 4),
        "reached_goal": reached_goal,
        "path": path,
        "steps": steps,
        "q_table": q_table,
        "eval_total_reward": sum(s["reward"] for s in steps)
    }