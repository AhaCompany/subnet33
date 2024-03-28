# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import time
import os
import hashlib

# Bittensor
import bittensor as bt

# Bittensor Validator Template:
import template
from template.validator import forward

# import base validator class which takes care of most of the boilerplate
from template.base.validator import BaseValidatorNeuron

import conversationgenome.utils
import conversationgenome.validator

from conversationgenome.ValidatorLib import ValidatorLib



class Validator(BaseValidatorNeuron):
    """
    Your validator neuron class. You should use this class to define your validator's behavior. In particular, you should replace the forward function with your own logic.

    This class inherits from the BaseValidatorNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a validator such as keeping a moving average of the scores of the miners and using them to set weights at the end of each epoch. Additionally, the scores are reset for new hotkeys at the end of each epoch.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()

        self.image_dir = './data/conversations/'
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

    async def forward(self):
        """
        Validator forward pass. Consists of:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """
        # TODO(developer): Rewrite this function based on your protocol definition.
        #return await forward(self)
        # get_random_uids is an example method, but you can replace it with your own.
        miner_uids = conversationgenome.utils.uids.get_random_uids(self, k=min(self.config.neuron.sample_size, self.metagraph.n.item()))
        print("miner_uids", miner_uids)
        vl = ValidatorLib()
        vl.validateMinimumTags([])

        # make a hash from the timestamp
        filename = hashlib.md5(str(time.time()).encode()).hexdigest()

        # Create a random image and load it.
        #image_data = ocr_subnet.validator.generate.invoice(path=os.path.join(self.image_dir, f"{filename}.pdf"), corrupt=True)

        # Create synapse object to send to the miner and attach the image.
        #baseImage = image_data['base64_image']
        baseImage = 21
        print("Send", baseImage)

        synapse = conversationgenome.protocol.Dummy(dummy_input = baseImage)

        rewards = None
        # The dendrite client queries the network.
        print("Query")
        responses = self.dendrite.query(
            # Send the query to selected miner axons in the network.
            axons=[self.metagraph.axons[uid] for uid in miner_uids],
            # Pass the synapse to the miner.
            synapse=synapse,
            # Do not deserialize the response so that we have access to the raw response.
            deserialize=False,
        )
        validResponses = []
        for response in responses:
            if not response.dummy_output:
                continue
            validResponses.append(response)
            bt.logging.info(f"CGP Received responses: {response.dummy_output}")
        # Log the results for monitoring purposes.
        responses = [{"tag":"baseball"},{"response":"Florida"}]
        #bt.logging.info(f"CGP Received responses: {responses}")
        #labels = image_data['labels']
        labels = ["Hello", "World"]

        rewards = conversationgenome.validator.reward.get_rewards(self, labels=labels, responses=responses)

        bt.logging.info(f"CGP Scored responses: {rewards}")

        # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
        #self.update_scores(rewards, miner_uids)

# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info("CGP Validator running...", time.time())
            time.sleep(5)
