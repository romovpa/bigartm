# This example implements online algorithm where topic model is updated several times
# during each scan of the collection.

import artm.messages_pb2, artm.library, sys, glob, os

# Parse collection
data_folder = sys.argv[1] if (len(sys.argv) >= 2) else ''
batches_disk_path = 'kos'
unique_tokens = artm.library.Library().load_dictionary(os.path.join(batches_disk_path, 'dictionary'))

# Create master component and infer topic model
with artm.library.MasterComponent() as master:
    master.get_config().processors_count = 2
    master.reconfigure()
    dictionary = master.CreateDictionary(unique_tokens)

    perplexity_score = master.CreatePerplexityScore()
    sparsity_theta_score = master.CreateSparsityThetaScore()
    sparsity_phi_score = master.CreateSparsityPhiScore()
    top_tokens_score = master.CreateTopTokensScore()
    theta_snippet_score = master.CreateThetaSnippetScore()

    # Configure basic regularizers
    theta_regularizer = master.CreateSmoothSparseThetaRegularizer()
    phi_regularizer = master.CreateSmoothSparsePhiRegularizer()
    decorrelator_regularizer = master.CreateDecorrelatorPhiRegularizer()

    # Configure the model
    model = master.create_model(topics_count=10, inner_iterations_count=10)
    model.enable_score(perplexity_score)
    model.enable_score(sparsity_phi_score)
    model.enable_score(sparsity_theta_score)
    model.enable_score(top_tokens_score)
    model.enable_score(theta_snippet_score)
    model.enable_regularizer(theta_regularizer, -0.1)
    model.enable_regularizer(phi_regularizer, -0.2)
    model.enable_regularizer(decorrelator_regularizer, 1000000)
    model.initialize(dictionary)  # Setup initial approximation for Phi matrix.

    # Online algorithm with AddBatch()
    update_every = 4
    batches = glob.glob(batches_disk_path + "/*.batch")
    for batch_index, batch_filename in enumerate(batches):
        master.add_batch(batch_filename=batch_filename)
        if ((batch_index + 1) % update_every == 0) or ((batch_index + 1) == len(batches)):
            master.wait_idle()  # wait for all batches are processed
            model.synchronize(decay_weight=0.9, apply_weight=0.1)  # synchronize model
            print "Perplexity = %.3f" % perplexity_score.get_value(model).value,
            print ", Phi sparsity = %.3f " % sparsity_phi_score.get_value(model).value,
            print ", Theta sparsity = %.3f" % sparsity_theta_score.get_value(model).value

    artm.library.Visualizers.print_top_tokens_score(top_tokens_score.get_value(model))
    artm.library.Visualizers.print_theta_snippet_score(theta_snippet_score.get_value(model))
