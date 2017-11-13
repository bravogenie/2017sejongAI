from nltk.corpus import movie_reviews 
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy
 
# Extract features from the input list of words
def extract_features(words):
    return dict([(word, True) for word in words])
 
if __name__=='__main__':
    # Load the reviews from the corpus 
    fileids_pos = movie_reviews.fileids('pos')
    fileids_neg = movie_reviews.fileids('neg')
     
    # Extract the features from the reviews
    features_pos = [(extract_features(movie_reviews.words(
            fileids=[f])), 'Positive') for f in fileids_pos]
    features_neg = [(extract_features(movie_reviews.words(
            fileids=[f])), 'Negative') for f in fileids_neg]
     
    # Define the train and test split (80% and 20%)
    threshold = 0.8
    num_pos = int(threshold * len(features_pos))
    num_neg = int(threshold * len(features_neg))
     
     # Create training and training datasets
    features_train = features_pos[:num_pos] + features_neg[:num_neg]
    features_test = features_pos[num_pos:] + features_neg[num_neg:]  

    # Print the number of datapoints used
    print('\nNumber of training datapoints:', len(features_train))
    print('Number of test datapoints:', len(features_test))
     
    # Train a Naive Bayes classifier 
    classifier = NaiveBayesClassifier.train(features_train)
    print('\nAccuracy of the classifier:', nltk_accuracy(
            classifier, features_test))

    N = 10
    print('\nTop ' + str(N) + ' most informative words:')
    for i, item in enumerate(classifier.most_informative_features()):
        print(str(i+1) + '. ' + item[0])
        if i == N - 1:
            break

    # Test input movie reviews
    input_reviews = [
        'The whole plot of the movie was a mess. It really confused me what all the characters in the film was fighting for, I mean there was no real goal of the movie for the first 70% of the time. Then they set up the story in a way where the friend turns into the foe, and those friends are now facing this foe as their primary enemy and not the one it was shown at the beginning of the movie. I think the whole script was written by a total non professional guy who needs to again go through the basic levels of a film school. This movie made DC lose its fame towards its fans and I hope they bounce back in their next show. Thank you', 
        'For the budget and start power... This was a total waste of time.',
        'I was really looking forward to this movie and only ended up with a huge deception.', 
        'I am really really disappointed to be honest..',
        'Overall, I will still watch DC movies, but I do feel like they are missing something that the Marvel counterpart has got right. Maybe it heart, maybe Marvel just have more practice but I hope DC can redeem themselves next year.',
        'This movie tries to be so much but ends up being a real mess. Where do I begin? Character introduction runs for almost 40 minutes and I still dont end up giving a damn about anyone. Jokes are often very misplaced and feels extremely forced. Many scenes feels like they were made solely for sake of the trailer and also the trailer resembles nothing of what the actual movie is like.',
        'SUICIDE SQUAD - YOU GOTTA BE KIDDING ME! THE MOVIE is A HOAX! It has been more than a month since I have watched this...this...tragic display of editing problems...this...potential hit turned to crap...this...brutal waste of reel and...let me tell ya...I am still shocked! There are so many things gone so wrong with Suicide Squad it seems almost decent at times.',
        'Lets be honest, it was just a bad movie. Jared Leto Joker may have been the worst ever. How do you screw this role up? He played the Joker as some scummy wannabe rapper. Unbelievable stuff. This is the sort of thing that had to come out of a corporate focus group.'

    ]

    print("\nMovie review predictions:")
    for review in input_reviews:
        print("\nReview:", review)

        # Compute the probabilities
        probabilities = classifier.prob_classify(extract_features(review.split()))

        # Pick the maximum value
        predicted_sentiment = probabilities.max()

        # Print outputs
        print("Predicted sentiment:", predicted_sentiment)
        print("Probability:", round(probabilities.prob(predicted_sentiment), 2))

