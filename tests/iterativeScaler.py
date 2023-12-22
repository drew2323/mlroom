"""
Tato cast pro iterativni scaling day by day se muze pouzit v sekvenceru

zatim byla odstraneno, budou prozkoumany zpusoby celkoveho scalingu
"""


            print("tady jsou vsechny features pro dany vstup")
            print("sem by sel iterativni scaler za tento den")
            print(features)

            # Transpose the data so that we have samples as rows and features as columns
            features = np.array(features)
            features = features.T

            scaler = StandardScaler()
            # Fitting the scaler to the data and transforming the data
            features = scaler.fit_transform(features)
            print("scaler vidi")
            print("pocet featur:", scaler.n_features_in_)
            #print(scaler.feature_names_in_)
            print("pocet samplu:", scaler.n_samples_seen_)

            # Transpose Back
            features = features.T
            features = features.tolist()