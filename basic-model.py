
    
import numpy as np
import random 

import matplotlib.pyplot as plt


# # '''new code try again'''
class Environment:
    def __init__(self):
        self.choice_values = [3,0,1,0,0,0,0,2]
        self.choices_provided = ['A','B','C','D','E','F','G','H'] 
        self.counter = np.random.randint(20, 30)

    def make_choices(self):
        print("Choices provided:", self.choices_provided)
        return self.choices_provided

    def reward_received(self,action):
        print(f'Action input: {action}')
        self.counter -= 1
        reversal_happened = False
        print(f"this is reversal counter is: {self.counter}")
        if self.counter == 0:
            self.reversal()
            self.counter = np.random.randint(20, 30)
            reversal_happened = True

        reward = self.choice_values[action]
        print(f"this is the action input: {action}, reward: {reward}")
        return action, reward, reversal_happened
    
    def reversal(self):
        #sort the list choice_values by the indexes of the values stored in the index
        ranked_indices = sorted(range(len(self.choice_values)), key=lambda i: self.choice_values[i], reverse=True)
        print(f'these are the rank_indices{ranked_indices}')

        best_idx = ranked_indices[0] #the index of the best value 
        sec_idx = ranked_indices[1]
        third_idx = ranked_indices[2]

        # Save initial values before modifying anything
        best_val = self.choice_values[best_idx] #values of the best and their position (index)
        sec_val = self.choice_values[sec_idx]
        third_val = self.choice_values[third_idx]

        # Rule 1: Second best choice gets the best value
        self.choice_values[sec_idx] = best_val

        # Rule 2: Best and Third-best choices randomly split second and third values
        self.choice_values[best_idx], self.choice_values[third_idx] = random.choice([
        (sec_val, third_val),
        (third_val, sec_val)
    ])
        print(f'this is post reversal choice_values {self.choice_values}')

    


class Agent: 

    def __init__(self):
        self.rank_choice_neurons = np.array([
                #col for choices (also can be index)
                [1,0,0,0,0,0,0,0],#row for best 
                [0,0,0,0,0,0,0,1],#row for second 
                [0,0,1,0,0,0,0,0] #row for third  
            ])
        self.value_rank =  np.array([
                #col for reward value: 0,1,2,3
                [0,0,0,1], #row for best 
                [0,0,1,0], #row for second 
                [0,1,0,0] #row for third 
            ]) 
        self.record = []

    def take_action(self):
        presumed_best_action = np.argmax(self.rank_choice_neurons[0])
        print(f'this is the presumed best action {presumed_best_action}')
        return presumed_best_action

    def update_state(self, chosen_action_index, reward):
        #which row (best,second,third) lies the choice 
        acting_rank = np.argmax(self.rank_choice_neurons[:, chosen_action_index]) #where is the action choice?
        print(f'this is the acting rank of chosen {acting_rank}')

        #add the choice and reward pair into the record 

        self.record.append((int(chosen_action_index),reward))
        print(f'this is the self.record: {self.record}')

        #find the row (rank) has a specific value (for example, which rank has the value of the reward?)
        reward_column = np.where(self.value_rank[:,reward]==1)[0]
        reward_col_index = reward_column[0]
        print(f'this is the reward column {reward_column}')

        if reward_col_index.any():
            holder = int(np.argmax(reward_column))
            if reward_col_index != acting_rank:
                self.value_rank[[reward_col_index,acting_rank]] = self.value_rank[[acting_rank,reward_col_index]]
                print(f'switch in value_rank has occured. {self.value_rank}')

                #update the choice_rank matrix based on the value_Rank matrix

        
        list_of_values_indices = np.argmax(self.value_rank, axis=1)
        print(f'the list of value indices  :{list_of_values_indices}')

        #sort the list by values corresponding to indices
        sorted_indices = np.argsort(-(list_of_values_indices))
        print(f'sorted values are {sorted_indices}')

        self.value_rank = self.value_rank[sorted_indices]
        self.rank_choice_neurons = self.rank_choice_neurons[sorted_indices]

        print(f'after sorting value rank: {self.value_rank}')
        print(f'after sorting choice rank: {self.rank_choice_neurons}')

        return self.record
       

#initialising agent and environment 
Env = Environment()
print(Env)

print(Env.choice_values)

agent1 = Agent()
print(agent1)

print(agent1.rank_choice_neurons)

def agent_action(agent,env):
    presumed_best_action = agent.take_action()
    action,reward,reversal_happened = env.reward_received(presumed_best_action)
    update_ranks = agent1.update_state(action,reward)
    actual_reward_choices_env = env.choice_values
    print(f'this is actual_reward_choices_env: {actual_reward_choices_env}')

    return action, reward, reversal_happened

'''plotting is a pain'''
magnitude_history = []
reversal_trials = []
choice_tower_0 = []
choice_tower_2 = []
choice_tower_7 = []

for trials in range(201):
    action, reward, reversal_happened = agent_action(agent1, Env)
    magnitude_history.append(list(Env.choice_values))
    if reversal_happened:
        reversal_trials.append(trials)


print(f'this is magnitude_history {magnitude_history}') 
print(f'this is reversal_happend: {reversal_happened}')
print(f'this is reversal trials: {reversal_trials}')

recorded_action, recorded_reward = zip(*agent1.record)

recorded_action = list(recorded_action)
print(f'this is recorded_action: {recorded_action}')

recorded_reward = list(recorded_reward)
print(f'this is recorded_reward: {recorded_reward}')

trials = range(1,len(recorded_reward)+1)
print(f'this is trials: {trials} ')




#keep track of choice_tower per trial

for sublist in magnitude_history:
    choice_tower_0.append(sublist[0])
    choice_tower_2.append(sublist[2])
    choice_tower_7.append(sublist[7])

print(choice_tower_0,choice_tower_2,choice_tower_7)

for i, rev in enumerate(reversal_trials):
    plt.axvline(
        x=rev,
        color='#424c51',
        linestyle='--',
        label='Reversal'if i==0 else '',
    )

#plot the action and reward history 

plt.plot(trials, choice_tower_0, color= '#B4BE89', label='Choice Tower 0')
plt.plot(trials, choice_tower_2, color= '#EB9191',label='Choice Tower 2')
plt.plot(trials, choice_tower_7, color='#749DA1',label='Choice Tower 7')

plt.ylabel('Reward Magnitude') 

plt.scatter(trials,recorded_reward,color='black',label='Choices')

plt.title('Basic-Model: Structured Bandit Task Session')
plt.xlabel('Trials')
plt.grid(True)
plt.legend()
plt.show()




        
    
    
    




        








    

    




    


                
            

