# This example demonstrates various ways of retrieving Theta matrix from BigARTM
# -*- coding: utf-8 -*-
import glob

import artm.messages_pb2
import artm.library

# Create master component and infer topic model
batches_disk_path = 'kos'
with artm.library.MasterComponent(disk_path=batches_disk_path) as master:
    master.get_config().cache_theta = True
    master.reconfigure()

    model = master.create_model(topics_count=8)
    theta_snippet_score = master.create_theta_snippet_score()
    model.enable_score(theta_snippet_score)

    for iteration in range(0, 2):
        master.invoke_iteration()
        master.wait_idle()  # wait for all batches are processed
        model.synchronize()  # synchronize model

    # Option 1.
    # Getting a small snippet of ThetaMatrix for last processed documents (just to get an impression how it looks)
    # This may be useful if you are debugging some weird behavior, playing with regularizer weights, etc.
    # This does not require "master.config().cache_theta = True"
    print "Option 1. ThetaSnippetScore."
    artm.library.Visualizers.print_theta_snippet_score(theta_snippet_score.get_value(model))

    # Option 2.
    # Getting a full theta matrix cached during last iteration
    # This does requires "master.config().cache_theta = True" and stores the entire Theta matrix in memory.
    theta_matrix = master.GetThetaMatrix(model, clean_cache=True)
    print "Option 2. Full ThetaMatrix cached during last iteration, #items = %i" % len(theta_matrix.item_id)

    # Option 3.
    # Getting theta matrix online during iteration.
    # This does requires "master.config().cache_theta = True", but never caches the entire Theta because we clean it.
    # This is the best alternative to Option 2 if you can not afford caching entire ThetaMatrix in memory.
    batches = glob.glob(batches_disk_path + "/*.batch")
    for batch_index, batch_filename in enumerate(batches):
        master.add_batch(batch_filename=batch_filename)

        # The following rule defines when to retrieve Theta matrix. You decide :)
        if ((batch_index + 1) % 2 == 0) or ((batch_index + 1) == len(batches)):
            master.wait_idle()  # wait for all batches are processed
            # model.Synchronize(decay_weight=..., apply_weight=...)  # uncomment for online algorithm
            theta_matrix = master.GetThetaMatrix(model=model, clean_cache=True)
            print "Option 3. ThetaMatrix from cache, online, #items = %i" % len(theta_matrix.item_id)

    # Option 4.
    # Testing batches by explicitly loading them from disk. This is the right way of testing held-out batches.
    # This does not require "master.config().cache_theta = True"
    test_batch = artm.library.Library().load_batch(batches[0])  # select the first batch for demo purpose
    theta_matrix = master.GetThetaMatrix(model=model, batch=test_batch)
    print "Option 4. ThetaMatrix for test batch, #items = %i" % len(theta_matrix.item_id)
