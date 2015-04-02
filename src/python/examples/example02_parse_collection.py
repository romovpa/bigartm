# This example parses a small text collection from disk and process it with BigARTM.
# Topic model is configured with sparsity regularizers (for theta and phi matrices),
# and also with topic decorrelator. The weight of all regularizers was adjusted manually
# and then hardcoded in this script.
# Several scores are printed on every iteration (perplexity score, sparsity of theta and phi matrix).

import artm.messages_pb2, artm.library, sys, glob

# Parse collection
data_folder = sys.argv[1] if (len(sys.argv) >= 2) else ''
target_folder = 'kos'
collection_name = 'kos'

# The following code is the same as library.ParseCollectionOrLoadDictionary(),
# but it is important for you to understand what it does.
# Please learn ParseCollection() and LoadDictionary() methods.

batches_found = len(glob.glob(target_folder + "/*.batch"))
if batches_found == 0:
  print "No batches found, parsing them from textual collection...",
  collection_parser_config = artm.messages_pb2.CollectionParserConfig();
  collection_parser_config.format = artm.library.CollectionParserConfig_Format_BagOfWordsUci

  collection_parser_config.docword_file_path = data_folder + 'docword.'+ collection_name + '.txt'
  collection_parser_config.vocab_file_path = data_folder + 'vocab.'+ collection_name + '.txt'
  collection_parser_config.target_folder = target_folder
  collection_parser_config.dictionary_file_name = 'dictionary'
  unique_tokens = artm.library.Library().parse_collection(collection_parser_config);
  print " OK."
else:
  print "Found " + str(batches_found) + " batches, using them."
  unique_tokens  = artm.library.Library().load_dictionary(target_folder + '/dictionary');

# Create master component and infer topic model
with artm.library.MasterComponent() as master:
  # Create dictionary with tokens frequencies
  dictionary           = master.CreateDictionary(unique_tokens)

  # Configure basic scores
  perplexity_score     = master.create_perplexity_score()
  sparsity_theta_score = master.create_sparsity_theta_score()
  sparsity_phi_score   = master.create_sparsity_phi_score()
  top_tokens_score     = master.create_top_tokens_score()
  theta_snippet_score  = master.create_theta_snippet_score()

  # Configure basic regularizers
  smsp_theta_reg   = master.create_smooth_sparse_theta_regularizer()
  smsp_phi_reg     = master.create_smooth_sparse_phi_regularizer()
  decorrelator_reg = master.create_decorrelator_phi_regularizer()

  # Configure the model
  model = master.create_model(topics_count = 10, inner_iterations_count = 10)
  model.enable_score(perplexity_score)
  model.enable_score(sparsity_phi_score)
  model.enable_score(sparsity_theta_score)
  model.enable_score(top_tokens_score)
  model.enable_score(theta_snippet_score)
  model.enable_regularizer(smsp_theta_reg, -0.1)
  model.enable_regularizer(smsp_phi_reg, -0.2)
  model.enable_regularizer(decorrelator_reg, 1000000)
  model.initialize(dictionary)       # Setup initial approximation for Phi matrix.

  for iter in range(0, 8):
    master.invoke_iteration(disk_path=target_folder)  # Invoke one scan over all batches,
    master.wait_idle();                               # and wait until it completes.
    model.synchronize();                             # Synchronize topic model.
    print "Iter#" + str(iter),
    print ": Perplexity = %.3f" % perplexity_score.get_value(model).value,
    print ", Phi sparsity = %.3f" % sparsity_phi_score.get_value(model).value,
    print ", Theta sparsity = %.3f" % sparsity_theta_score.get_value(model).value

  artm.library.Visualizers.print_top_tokens_score(top_tokens_score.get_value(model))
  artm.library.Visualizers.print_theta_snippet_score(theta_snippet_score.get_value(model))
