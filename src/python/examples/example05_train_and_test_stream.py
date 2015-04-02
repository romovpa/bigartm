# This example creates train (90%) and test (10%) streams in the collection.
# Train stream is used to tune topic model, and test stream is used to calculate perplexity.

import artm.messages_pb2, artm.library, sys

# Parse collection
data_folder = sys.argv[1] if (len(sys.argv) >= 2) else ''
target_folder = 'kos'
collection_name = 'kos'
unique_tokens = artm.library.Library().parse_collection_or_load_dictionary(
  data_folder + 'docword.'+ collection_name + '.txt',
  data_folder + 'vocab.' + collection_name + '.txt',
  target_folder)

with artm.library.MasterComponent() as master:
  dictionary = master.CreateDictionary(unique_tokens)
  # Configure test and train streams
  train_stream = artm.messages_pb2.Stream()
  train_stream.name = 'train_stream'
  train_stream.type = artm.library.Stream_Type_ItemIdModulus
  train_stream.modulus = 10
  for i in range(0, train_stream.modulus - 1):
    train_stream.residuals.append(i)
  master.create_stream(train_stream)

  test_stream = artm.messages_pb2.Stream()
  test_stream.name = 'test_stream'
  test_stream.type = artm.library.Stream_Type_ItemIdModulus
  test_stream.modulus = train_stream.modulus
  test_stream.residuals.append(test_stream.modulus - 1)
  master.create_stream(test_stream)

  perplexity_train_score = master.CreatePerplexityScore(stream_name = train_stream.name)
  perplexity_test_score  = master.CreatePerplexityScore(stream_name = test_stream.name)

  # Configure the model
  model = master.create_model(topics_count = 10, inner_iterations_count = 10)
  model.get_config().stream_name = train_stream.name
  model.enable_score(perplexity_train_score)
  model.enable_score(perplexity_test_score)
  model.initialize(dictionary)       # Setup initial approximation for Phi matrix.

  for iter in range(0, 8):
    master.invoke_iteration(disk_path=target_folder)  # Invoke one scan of the entire collection...
    master.wait_idle();                               # and wait until it completes.
    model.synchronize();                             # Synchronize topic model.
    print "Iter#" + str(iter),
    print ": Train perplexity = %.3f" % perplexity_train_score.get_value(model).value,
    print ", Test perplexity = %.3f " % perplexity_test_score.get_value(model).value
