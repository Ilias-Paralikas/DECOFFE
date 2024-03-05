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
        self.number_of_servers = number_of_servers
        
        if self.single_agent:
           self.agent = Agent(id =0,
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
                    dueling=dueling)
        else:
            self.agent_store_counter = 0
            self.agent_action_counter = 0
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
            for i in range(number_of_servers)]
            
            
    def store_transitions(self,
                           state,
                           lstm_state,
                           action,
                           reward,
                           new_state,
                           new_lstm_state,
                           done
                           ):
        if self.single_agent:
            self.agent.store_transitions(state = state,
                                                lstm_state=lstm_state,
                                                action = action,
                                                reward= reward,
                                                new_state=new_state,
                                                new_lstm_state=new_lstm_state,
                                                done=done)     
            self.agent.learn()       
            
        else:
            self.agent_store_counter =  self.agent_store_counter %self.number_of_servers
            self.agents[self.agent_store_counter].store_transitions(state = state,
                                                lstm_state=lstm_state,
                                                action = action,
                                                reward= reward,
                                                new_state=new_state,
                                                new_lstm_state=new_lstm_state,
                                                done=done) 
            self.agents[self.agent_store_counter].learn()           
            self.agent_store_counter +=1
    
    

    def choose_action(self,state,lstm_input):
        if self.single_agent:
            action = self.agent.choose_action(state,lstm_input)
        else:
            self.agent_action_counter =  self.agent_action_counter %self.number_of_servers
            action = self.agents[self.agent_action_counter].choose_action(state,lstm_input)
            self.agent_action_counter +=1
        return action

    
    
    def get_espilon(self):
        if self.single_agent:
            return self.agent.epsilon
        else:
            return self.agents[0].epsilon
