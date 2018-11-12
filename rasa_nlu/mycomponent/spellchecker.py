from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData

class SpellChecker(Component):
    """A new component"""

    # Name of the component to be used when integrating it in a
    # pipeline. E.g. ``[ComponentA, ComponentB]``
    # will be a proper pipeline definition where ``ComponentA``
    # is the name of the first component of the pipeline.
    name = "spell_checker"

    # Defines what attributes the pipeline component will
    # provide when called. The listed attributes
    # should be set by the component on the message object
    # during test and train, e.g.
    # ```message.set("entities", [...])```
    provides = ["checked_text"]

    # Which attributes on a message are required by this
    # component. e.g. if requires contains "tokens", than a
    # previous component in the pipeline needs to have "tokens"
    # within the above described `provides` property.
    
    #requires = []

    # Defines the default configuration parameters of a component
    # these values can be overwritten in the pipeline configuration
    # of the model. The component should choose sensible defaults
    # and should be able to create reasonable results with the defaults.
    defaults = {"model":None}

    # Defines what language(s) this component can handle.
    # This attribute is designed for instance method: `can_handle_language`.
    # Default value is None which means it can handle all languages.
    # This is an important feature for backwards compatibility of components.
    language_list = ["zh"]

    def __init__(self, component_config=None,checker = None):
        self.checker = checker
        super( SpellChecker, self).__init__(component_config)
        
    def process(self, message, **kwargs):
        """Process an incoming message.

        This is the components chance to process an incoming
        message. The component can rely on
        any context attribute to be present, that gets created
        by a call to :meth:`components.Component.pipeline_init`
        of ANY component and
        on any context attributes created by a call to
        :meth:`components.Component.process`
        of components previous to this one."""
        text = message.text
        checked_text = self.checker(text)[0]
#         try:
#             checked_text = self.checker(text)[0]
#         except:
#             checked_text = "None"
        #返回纠错后的句子
        message.set("checked_text",checked_text,add_to_output=True)
        #add_to_output True 后会添加到最后的结果中

    @classmethod
    def create(cls, cfg):
        # type: (RasaNLUModelConfig) -> JiebaTokenizer
        #模块的算法加载。
        import pycorrector
        
        component_conf = cfg.for_component(cls.name, cls.defaults)
        corrected_sent, detail = pycorrector.correct("少先队员因该为老人让坐")
        print("!!!spell checker loaded,",corrected_sent)
        checker = pycorrector.correct              
        return cls(component_conf, checker)        
