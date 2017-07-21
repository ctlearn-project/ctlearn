def createEmbeddings():
        #embeddings visualization stuff
        config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
        embedding_var = tf.Variable(tf.stack(embeddings_event_list,axis=0), name='images')
        
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        # Link this tensor to its metadata file (e.g. labels).
        embedding.metadata_path = os.path.join(args.logdir, 'metadata.tsv')

        #write metadata file with labels
        with open(embedding.metadata_path, 'w') as metadata_file:
            for i in range(NUM_IMAGES_VIZ):
                metadata_file.write('{}\n'.format(table.read(i,i+1,field='gamma_hadron_label')[0]))

        #add embeddings to summary writer
        projector.visualize_embeddings(sv.summary_writer, config)



