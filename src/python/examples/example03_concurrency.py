# This example prints out the average execution time per iteration
# depending on the number of concurrent processors.

import artm.messages_pb2, artm.library, sys, time

# Parse collection
data_folder = sys.argv[1] if (len(sys.argv) >= 2) else ''
target_folder = 'kos'
collection_name = 'kos'
unique_tokens = artm.library.Library().parse_collection_or_load_dictionary(
  data_folder + 'docword.'+ collection_name + '.txt',
  data_folder + 'vocab.' + collection_name + '.txt',
  target_folder)

# Create master component and infer topic model
for processors_count in [4,2,1]:
  with artm.library.MasterComponent() as master:
    dictionary = master.CreateDictionary(unique_tokens)

    perplexity_score = master.CreatePerplexityScore()
    model = master.create_model(topics_count = 10, inner_iterations_count = 10)
    model.enable_score(perplexity_score)
    model.initialize(dictionary)       # Setup initial approximation for Phi matrix.

    print "Setting processors_count to " + str(processors_count)
    master.get_config().processors_count = processors_count
    master.reconfigure()

    times = []
    num_iters = 5
    for iter in range(0, num_iters):
      start = time.time()
      master.invoke_iteration(disk_path=target_folder)  # Invoke one scan over all batches,
      master.wait_idle();                               # and wait until it completes.
      model.synchronize();                             # Synchronize topic model.
      end = time.time()
      times.append(end - start)
      print "Iter#" + str(iter),
      print ": Perplexity = %.3f" % perplexity_score.get_value(model).value + ", Time = %.3f" % (end - start)

    print "Averate time per iteration = %.3f " % (float(sum(times))/len(times)) + "\n"
