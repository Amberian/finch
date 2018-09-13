from tensorflow.python.ops import array_ops
from tensorflow.contrib.seq2seq.python.ops.basic_decoder import BasicDecoder
from tensorflow.contrib.seq2seq.python.ops.beam_search_decoder import BeamSearchDecoder


class ModifiedBasicDecoder(BasicDecoder):
    def __init__(self, cell, helper, initial_state, concat_z, output_layer=None):
        super().__init__(cell, helper, initial_state, output_layer)
        self.z = concat_z

    def initialize(self, name=None):
        (finished, first_inputs, initial_state) = super().initialize(name)
        first_inputs = array_ops.concat([first_inputs, self.z], -1)
        return (finished, first_inputs, initial_state)

    def step(self, time, inputs, state, name=None):
        (outputs, next_state, next_inputs, finished) = super().step(
            time, inputs, state, name)
        next_inputs = array_ops.concat([next_inputs, self.z], -1)
        return (outputs, next_state, next_inputs, finished)


class ModifiedBeamSearchDecoder(BeamSearchDecoder):
    def __init__(self,
                 cell,
                 embedding,
                 start_tokens,
                 end_token,
                 initial_state,
                 beam_width,
                 concat_z,
                 output_layer=None,
                 length_penalty_weight=0.0):
        super().__init__(cell, embedding, start_tokens, end_token, initial_state, beam_width, output_layer, length_penalty_weight)
        self.z = concat_z

    def initialize(self, name=None):
        (finished, start_inputs, initial_state) = super().initialize(name)
        start_inputs = array_ops.concat([start_inputs, self.z], -1)
        return (finished, start_inputs, initial_state)

    def step(self, time, inputs, state, name=None):
        (beam_search_output, beam_search_state, next_inputs, finished) = super().step(
            time, inputs, state, name)
        next_inputs = array_ops.concat([next_inputs, self.z], -1)
        return (beam_search_output, beam_search_state, next_inputs, finished)
