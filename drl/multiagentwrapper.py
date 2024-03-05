from .agent import Agent
import torch
class MultiAgentWrapper:
    def __init__(self,
                number_of_servers,
                state_dimensions,
                lstm_shape,
                number_of_actions,
                hidden_layers,
                lstm_layers,
                epsilon_decrement,
                batch_size,
                learning_rate,
                memory_size,
                lstm_time_step,
                replace_target_iter,
                optimizer,
                loss_function,
                device,
                checkpoint_folder,
                single_agent,
                epsilon,
                gamma,
                epsilon_end,
                local_action_probability,
                save_model_frequency ,
                read_checkpoint = True,
                dueling=True):
        
        self.single_agent = single_agent
        self.agent_store_counter = 0
        self.agent_action_counter = 0
        self.agent_learn_counter = 0
        if self.single_agent:
            self.number_of_agents = 1
            self.agents = [Agent(id =0,
                    state_dimensions=state_dimensions,
                    lstm_shape=lstm_shape,
                    number_of_actions=number_of_actions,
                    hidden_layers =hidden_layers,
                    lstm_layers = lstm_layers,
                    epsilon_decrement =epsilon_decrement,
                    batch_size =batch_size,
                    learning_rate =learning_rate,
                    memory_size = memory_size,
                    lstm_time_step = lstm_time_step,
                    replace_target_iter = replace_target_iter,
                    loss_function = getattr(torch.nn, loss_function),
                    optimizer = getattr(torch.optim, optimizer),
                    device=device,
                    checkpoint_folder=checkpoint_folder+'/single_agent.pt',
                    gamma=gamma,
                    epsilon=epsilon,
                    epsilon_end=epsilon_end,
                    local_action_probability = local_action_probability,
                    save_model_frequency = save_model_frequency,
                    read_checkpoint = read_checkpoint,
                    dueling=dueling)]
        else:
            self.number_of_agents = number_of_servers
            self.agents = [Agent(id =i,
                    state_dimensions=state_dimensions,
                    lstm_shape=lstm_shape,
                    number_of_actions=number_of_actions,
                    hidden_layers =hidden_layers,
                    lstm_layers = lstm_layers,
                    epsilon_decrement =epsilon_decrement,
                    batch_size =batch_size,
                    learning_rate =learning_rate,
                    memory_size = memory_size,
                    lstm_time_step = lstm_time_step,
                    replace_target_iter = replace_target_iter,
                    loss_function = getattr(torch.nn, loss_function),
                    optimizer = getattr(torch.optim, optimizer),
                    device=device,
                    checkpoint_folder=checkpoint_folder+'/agent_'+str(i)+'.pt' ,
                    gamma=gamma,
                    epsilon=epsilon,
                    epsilon_end=epsilon_end,
                    local_action_probability = local_action_probability,
                    save_model_frequency = save_model_frequency,
                    read_checkpoint = read_checkpoint,
                    dueling=dueling)
            for i in range(self.number_of_agents)]
            
            
    def store_transitions(self,
                           state,
                           lstm_state,
                           action,
                           reward,
                           new_state,
                           new_lstm_state,
                           done
                           ):

        self.agent_store_counter =  self.agent_store_counter %self.number_of_agents
        self.agents[self.agent_store_counter].store_transitions(state = state,
                                            lstm_state=lstm_state,
                                            action = action,
                                            reward= reward,
                                            new_state=new_state,
                                            new_lstm_state=new_lstm_state,
                                            done=done) 
        self.agent_store_counter +=1
    
    
    def learn(self):
        self.agent_learn_counter = self.agent_learn_counter %self.number_of_agents
        self.agents[self.agent_learn_counter].learn()           
        self.agent_learn_counter+=1
    

    def choose_action(self,state,lstm_input):
        self.agent_action_counter =  self.agent_action_counter %self.number_of_agents
        action = self.agents[self.agent_action_counter].choose_action(state,lstm_input)
        self.agent_action_counter +=1
        return action

    
    
    def get_espilon(self):
        return self.agents[0].epsilon
