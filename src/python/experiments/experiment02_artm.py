from __future__ import division

import os
import sys
import time
import math

home_folder = '/home/ubuntu/'
sys.path.append(home_folder + 'bigartm/src/python')
sys.path.append(home_folder + 'bigartm/src/python/artm')

import artm.messages_pb2
import artm.library

#######################################################################################################################

online_timeout          = 10     # option of frequency of updates in online mode
processors_count        = 1      # number of Processor threads to be used in experiment

# number of iterations over whole collection. Note, that n + 1 means n iteration + scanning of first
# little part of collection for qualification of results 
outer_iterations_count  = 1 + 1

inner_iterations_count  = 10     # number of iteration over each document
topics_count            = 100    # number of topics we need to extract from collection
kappa                   = 0.5    # parameter for coefficient of forgetting between sync-s
tau0                    = 64     # parameter for coefficient of forgetting between sync-s
batch_size              = 10000  # size of batches in documents
update_every            = 1      # how many batches to process before next synchronization
save_and_test_model     = False  # save topic model into file for future usage

# parameter of topic kernels, tokens with p(t|w) >= this value would form the kernels
probability_threshold   = 0.25

# path with batches
batches_disk_path       = home_folder + 'wiki_10k'

# full filename of file with dictionary
dictionary_file         = home_folder + 'dictionary'

# name of test batch for held-out perplexity. Set 'use_test_batch' flag to 'False', if there's no
# need in test. 'test_every' is the number of sync-s before next counting of hold-out perplexity
test_batch_name         = home_folder + '243af5b8-beab-4332-bb42-61892df5b044.batch'
test_every              = 20
use_test_batch          = True

# path with batches for final held-out estimation
test_batches_folder     = home_folder + 'test'

# number of documents be processed without regularization
first_documents         = 70000

# number of documents be re-processed after last iteration
last_documents         = 80000

# tau coefficients for ARTM
tau_decor               = 5.8e+5
tau_phi                 = -0.011
tau_theta               = -0.15

#######################################################################################################################

if (not os.path.isdir('results')): 
  os.mkdir('results')
os.chdir('results')

# open files for information about scores on each outer iteration
perplexity_file            = open('perplexity.txt', 'w')
theta_sparsity_file        = open('theta_sparsity.txt', 'w')
phi_sparsity_file          = open('phi_sparsity.txt', 'w')
topic_kernel_size_file     = open('topic_kernel_size.txt', 'w')
topic_kernel_purity_file   = open('topic_kernel_purity.txt', 'w')
topic_kernel_contrast_file = open('topic_kernel_contrast.txt', 'w')

# put log-files in specified folder
os.chdir('..')
if (not os.path.isdir('log')):
  os.mkdir('log')
os.chdir('log')

#######################################################################################################################

# create the configuration of Master
master_config                  = artm.messages_pb2.MasterComponentConfig()
master_config.disk_path        = batches_disk_path
master_config.processors_count = processors_count

# read static dictionary message with information about collection
dictionary_message = artm.library.Library().load_dictionary(dictionary_file)
test_batch = artm.library.Library().load_batch(test_batch_name)

with artm.library.MasterComponent(master_config) as master:
  print 'Technical tasks and loading model initialization...'
  # create static dictionary in Master
  dictionary = master.CreateDictionary(dictionary_message)

  # create and configure scores in Master
  perplexity_score = master.create_perplexity_score()
  sparsity_theta_score = master.create_sparsity_theta_score()
  sparsity_phi_score = master.create_sparsity_phi_score()

  topic_kernel_score_config = artm.messages_pb2.TopicKernelScoreConfig()
  topic_kernel_score_config.probability_mass_threshold = probability_threshold
  topic_kernel_score = master.create_topic_kernel_score(config = topic_kernel_score_config)
  
  items_processed_score = master.create_items_processed_score()


  # create and configure regularizers in Master
  decorrelator_reg = master.create_decorrelator_phi_regularizer()
  sparse_phi_reg = master.create_smooth_sparse_phi_regularizer()
  sparse_theta_reg = master.create_smooth_sparse_theta_regularizer()

  # create configuration of Model
  model_config                        = artm.messages_pb2.ModelConfig()
  model_config.topics_count           = topics_count
  model_config.inner_iterations_count = inner_iterations_count

  # create Model according to its configuration
  model = master.create_model(model_config)

  # enable scores in the Model
  model.enable_score(perplexity_score)
  model.enable_score(sparsity_theta_score)
  model.enable_score(sparsity_phi_score)
  model.enable_score(topic_kernel_score)
  model.enable_score(items_processed_score)

  # enable regularizes in Model and initialize them with 0 tau_coefficients
  model.enable_regularizer(decorrelator_reg, 0)
  model.enable_regularizer(sparse_phi_reg, 0)
  model.enable_regularizer(sparse_theta_reg, 0)
  
  # set initial approximation for Phi matrix
  model.initialize(dictionary)

#######################################################################################################################

  # global time counter:
  elapsed_time = 0.0
  
  # number of documents, found in collection
  max_items = 0
  
  first_sync = True
  need_to_update = True
  # start collection processing
  print '\n=======Experiment was started=======\n'
  for outer_iteration in range(0, outer_iterations_count):
    start_time = time.clock()
    sync_count = -1
    # invoke one scan of the whole collection
    master.invoke_iteration(1)

    done = False
    next_items_processed = batch_size * update_every

    test_on_this_iter = 0
    while (not done):
      online_start_time = time.clock()
      # Wait 'online_timeout' ms and check if the number of processed items had changed
      done = master.wait_idle(online_timeout)
      current_items_processed = items_processed_score.get_value(model).value
      if done or (current_items_processed >= next_items_processed):
        sync_count += 1
        update_coef = current_items_processed / (batch_size * update_every)
        next_items_processed = current_items_processed + (batch_size * update_every)      # set next model update
        rho = pow(tau0 + update_coef, -kappa)                                             # calculate rho
        model.synchronize(decay_weight=(0 if first_sync else (1-rho)), apply_weight=rho)  # synchronize model
        first_sync = False

        # update tau_coefficients of regularizers in Model
        if (need_to_update and (next_items_processed >= first_documents) and (outer_iteration == 0)):
          config_copy = artm.messages_pb2.ModelConfig()
          config_copy.CopyFrom(model.get_config())
          config_copy.regularizer_tau[0] = tau_decor
          config_copy.regularizer_tau[1] = tau_phi
          config_copy.regularizer_tau[2] = tau_theta
          model.reconfigure(config_copy)
          need_to_update = False

        # get current scores values
        sparsity_phi_score_value    = sparsity_phi_score.get_value(model).value
        sparsity_theta_score_value  = sparsity_theta_score.get_value(model).value
        topic_kernel_score_value    = topic_kernel_score.get_value(model)
        items_processed_score_value = items_processed_score.get_value(model).value

        perplexity_score_value = -1
        if (test_on_this_iter % test_every == 0) or\
           (current_items_processed > last_documents and outer_iteration == outer_iterations_count - 1):
          perplexity_score_value = perplexity_score.get_value(model = model, batch = test_batch).value
        test_on_this_iter += 1

        # increase time counter and save iteration time
        iteration_time = time.clock() - online_start_time
        elapsed_time += iteration_time

        # display information into terminal
        print '=========================================================='
        print 'Synchronization #' + '%2s' % str(sync_count) +\
              '      | perplexity = ' + '%6s' %\
              (str(round(perplexity_score_value)) if (perplexity_score_value != -1) else 'NO')
        print '----------------------------------------------------------'
        print 'Phi sparsity = ' + '%7s' % str(round(sparsity_phi_score_value, 4) * 100) +\
              ' % | ' + 'Theta sparsity = ' + '%7s' %\
              str(round(sparsity_theta_score_value, 4) * 100) + ' %'
        print '----------------------------------------------------------'
        print 'Size = ' + '%7s' % str(round(topic_kernel_score_value.average_kernel_size)) + '  |  ' +\
              'Purity = ' + '%7s' % str(round(topic_kernel_score_value.average_kernel_purity, 3)) + '  |  ' +\
              'Contrast = ' + '%7s' % str(round(topic_kernel_score_value.average_kernel_contrast, 3))
        print '----------------------------------------------------------'      
        print 'Elapsed time = ' + '%7s' % str(round(iteration_time, 2)) + ' sec.' + ' | ' +\
              'Items processed = ' + '%10s' % str(items_processed_score_value)
        print '==========================================================\n\n'

        # update current max documents count
        if (items_processed_score_value > max_items):
          max_items = items_processed_score_value
        else:
          items_processed_score_value += max_items * outer_iteration
          
        # put information into corresponding files
        if (perplexity_score_value > 0):
          perplexity_file.write('(' + str(items_processed_score_value) +\
              ', ' + str(round(perplexity_score_value)) + ')\n')
          phi_sparsity_file.write('(' + str(items_processed_score_value) +\
              ', ' +   str(round(sparsity_phi_score_value, 4) * 100) + ')\n')
          theta_sparsity_file.write('(' + str(items_processed_score_value) +\
              ', ' + str(round(sparsity_theta_score_value, 4) * 100) + ')\n')
          topic_kernel_size_file.write('(' + str(items_processed_score_value) +\
              ', ' + str(round(topic_kernel_score_value.average_kernel_size)) + ')\n')
          topic_kernel_purity_file.write('(' + str(items_processed_score_value) +\
              ', ' + str(round(topic_kernel_score_value.average_kernel_purity, 3)) + ')\n')
          topic_kernel_contrast_file.write('(' + str(items_processed_score_value) +\
              ', ' + str(round(topic_kernel_score_value.average_kernel_contrast, 3)) + ')\n')

        if ((current_items_processed > last_documents) and (outer_iteration == outer_iterations_count - 1)):
          print 'All elapsed time = ' + str(elapsed_time)
          if (save_and_test_model):
            print 'Saving topic model... ',
            with open(home_folder + 'Output.topic_model', 'wb') as binary_file:
              binary_file.write(master.GetTopicModel(model).SerializeToString())
          break
          
# close all opened files and finish the program
perplexity_file.close()
theta_sparsity_file.close()
phi_sparsity_file.close()
topic_kernel_size_file.close()
topic_kernel_purity_file.close()
topic_kernel_contrast_file.close()

if (save_and_test_model):
  processors_count = 8     # change number of processors to increase speed
  
  # create the configuration of Master
  test_master_config                  = artm.messages_pb2.MasterComponentConfig()
  test_master_config.processors_count = processors_count
  test_master_config.disk_path        = test_batches_folder
  
  with artm.library.MasterComponent(test_master_config) as test_master:
    # read saved topic model from file
    print 'Loading topic model...\n',
    topic_model = artm.messages_pb2.TopicModel()
    with open(home_folder + 'Output.topic_model', 'rb') as binary_file:
        topic_model.ParseFromString(binary_file.read())

    # create static dictionary in Master
    test_dictionary = test_master.CreateDictionary(dictionary_message)
    # create perplexity score in Master
    test_perplexity_score = test_master.create_perplexity_score()
    
    # Create model for testing and enable perplexity scoring in it
    test_model = test_master.create_model(topics_count = topics_count, inner_iterations_count = inner_iterations_count)
    test_model.enable_score(test_perplexity_score)
    
    # restore previously saved topic model into test_master
    test_model.overwrite(topic_model)

    # process batches, count perplexity and display the result
    print 'Estimate perplexity on held-out batches...\n'
    test_master.invoke_iteration()
    test_master.wait_idle()
    print "Test Perplexity calculated in BigARTM = %.3f" % test_perplexity_score.get_value(test_model).value
print '\n=======Experiment was finished=======\n'
