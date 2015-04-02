# This example demonstrates objective topics (with high sparsity) and background topics (without sparsity)

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

    background_topics = []
    objective_topics = []
    all_topics = []

    for i in range(0, 16):
        topic_name = "topic" + str(i)
        all_topics.append(topic_name)
        if i < 14:
            objective_topics.append(topic_name)
        else:
            background_topics.append(topic_name)

    perplexity_score = master.create_perplexity_score()
    sparsity_theta_objective = master.create_sparsity_theta_score(topic_names=objective_topics)
    sparsity_phi_objective = master.create_sparsity_phi_score(topic_names=objective_topics)
    top_tokens_score = master.create_top_tokens_score()
    theta_snippet_score = master.create_theta_snippet_score()

    # Configure basic regularizers
    theta_objective = master.create_smooth_sparse_theta_regularizer(topic_names=objective_topics)
    theta_background = master.create_smooth_sparse_theta_regularizer(topic_names=background_topics)
    phi_objective = master.create_smooth_sparse_phi_regularizer(topic_names=objective_topics)
    phi_background = master.create_smooth_sparse_phi_regularizer(topic_names=background_topics)
    decorrelator_regularizer = master.create_decorrelator_phi_regularizer(topic_names=objective_topics)

    # Configure the model
    model = master.create_model(topics_count=10, inner_iterations_count=30, topic_names=all_topics)
    model.enable_score(perplexity_score)
    model.enable_score(sparsity_theta_objective)
    model.enable_score(sparsity_phi_objective)
    model.enable_score(top_tokens_score)
    model.enable_score(theta_snippet_score)
    model.enable_regularizer(theta_objective, -1.0)
    model.enable_regularizer(theta_background, 0.5)
    model.enable_regularizer(phi_objective, -1.0)
    model.enable_regularizer(phi_background, 1.0)
    model.enable_regularizer(decorrelator_regularizer, 1000000)
    model.initialize(dictionary)  # Setup initial approximation for Phi matrix.

    # Online algorithm with AddBatch()
    update_every = master.get_config().processors_count
    batches = glob.glob(batches_disk_path + "/*.batch")

    for iteration in range(0, 5):
        for batch_index, batch_filename in enumerate(batches):
            master.add_batch(batch_filename=batch_filename)
            if ((batch_index + 1) % update_every == 0) or ((batch_index + 1) == len(batches)):
                master.wait_idle()  # wait for all batches are processed
                model.synchronize(decay_weight=0.9, apply_weight=0.1)  # synchronize model
                print "Perplexity = %.3f" % perplexity_score.get_value(model).value,
                print ", Phi objective sparsity = %.3f" % sparsity_phi_objective.get_value(model).value,
                print ", Theta objective sparsity = %.3f" % sparsity_theta_objective.get_value(model).value

    artm.library.Visualizers.print_top_tokens_score(top_tokens_score.get_value(model))
    artm.library.Visualizers.print_theta_snippet_score(theta_snippet_score.get_value(model))
