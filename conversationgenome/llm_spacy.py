import json

spacy = None
Matcher = None
try:
    import spacy
    from spacy.matcher import Matcher
except:
    print("Please install spacy to run locally")
    # en_core_web_sm model vectors = 96 dimensions.
    # en_core_web_md and en_core_web_lg = 300 dimensions
    # python -m spacy download en_core_web_sm
    # python -m spacy download en_core_web_lg



class llm_spacy:
    nlp = None

    def convert(self):
        print("Convert Spacy")

    def get_nlp(self):
        nlp = self.nlp
        dataset = "en_core_web_lg"
        if not nlp:
            # python -m spacy download en_core_web_sm
            #nlp = spacy.load("en_core_web_sm")
            #nlp = spacy.load("en_core_web_md")

            if not spacy.util.is_package(dataset):
                # download model
                spacy.cli.download(dataset)
                print("Model downloaded successfully!")

            nlp = spacy.load(dataset) # ~600mb
            #print(f"Vector dimensionality: {nlp.vocab.vectors_length}")
            self.nlp = nlp
        return nlp

    async def simple_text_to_tags(self, body):
        nlp = self.get_nlp()

        # Define patterns
        adj_noun_pattern = [{"POS": "ADJ"}, {"POS": "NOUN"}]
        pronoun_pattern = [{"POS": "PRON"}]
        unique_word_pattern = [{"POS": {"IN": ["NOUN", "VERB", "ADJ"]}, "IS_STOP": False}]

        # Initialize the Matcher with the shared vocabulary
        matcher = Matcher(nlp.vocab)
        matcher.add("ADJ_NOUN_PATTERN", [adj_noun_pattern])
        matcher.add("PRONOUN_PATTERN", [pronoun_pattern])
        matcher.add("UNIQUE_WORD_PATTERN", [unique_word_pattern])

        doc = nlp( body )
        #print("DOC", doc)
        matches = matcher(doc)
        matches_dict = {}
        for match_id, start, end in matches:
            span = doc[start:end]
            #matchPhrase = span.text
            matchPhrase = span.lemma_
            if len(matchPhrase) > 5:
                #print(f"Original: {span.text}, Lemma: {span.lemma_} Vectors: {span.vector.tolist()}")
                if not matchPhrase in matches_dict:
                    matches_dict[matchPhrase] = {"tag":matchPhrase, "count":0, "vectors":span.vector.tolist()}
                matches_dict[matchPhrase]['count'] += 1

        return matches_dict


    async def test_neighborhood(self):
        print("STRT test_neighborhood")
        nlp = self.get_nlp()

        # Process the content and the individual tag
        content = "I love playing football and basketball. Sports are a great way to stay active."
        tag = "sports"
        content_doc = nlp(content)
        tag_doc = nlp(tag)

        allVectors = [token.vector for token in content_doc]
        #print("allVectors",allVectors )
        # Create a vector representing the entire content by averaging the vectors of all tokens
        content_vector = np.mean(allVectors, axis=0)
        #print("content_vector", content_vector)

        # Calculate the similarity score between the content vector and the tag vector
        tag_vector = tag_doc[0].vector
        similarity_score = np.dot(content_vector, tag_vector) / (np.linalg.norm(content_vector) * np.linalg.norm(tag_vector))
        print(f"Similarity score between the content and the tag '{tag}': {similarity_score}")

    async def conversation_to_metadata(self,  convo):
        # For this simple matcher, just munge all of the lines together
        body = json.dumps(convo['lines'])
        matches_dict = await self.simple_text_to_tags(body)
        return {"tags": matches_dict}

