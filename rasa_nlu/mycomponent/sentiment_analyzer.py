from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData

class SentimentAnalyzer(Component):
    """A new component"""

    # Name of the component to be used when integrating it in a
    # pipeline. E.g. ``[ComponentA, ComponentB]``
    # will be a proper pipeline definition where ``ComponentA``
    # is the name of the first component of the pipeline.
    name = "sentiment_analyzer"

    # Defines what attributes the pipeline component will
    # provide when called. The listed attributes
    # should be set by the component on the message object
    # during test and train, e.g.
    # ```message.set("entities", [...])```
    provides = ["sentiment_score"]

    # Which attributes on a message are required by this
    # component. e.g. if requires contains "tokens", than a
    # previous component in the pipeline needs to have "tokens"
    # within the above described `provides` property.
    
    #requires = []

    # Defines the default configuration parameters of a component
    # these values can be overwritten in the pipeline configuration
    # of the model. The component should choose sensible defaults
    # and should be able to create reasonable results with the defaults.
    defaults = {"stopwords":None}

    # Defines what language(s) this component can handle.
    # This attribute is designed for instance method: `can_handle_language`.
    # Default value is None which means it can handle all languages.
    # This is an important feature for backwards compatibility of components.
    language_list = ["zh"]

    def __init__(self, component_config=None,analyzer = None):
        self.analyzer = analyzer
        super(SentimentAnalyzer, self).__init__(component_config)
        
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
        sentiment_score = self.analyzer(text)
        #返回纠错后的句子
        message.set("sentiment_score",sentiment_score,add_to_output=True)
        #add_to_output True 后会添加到最后的结果中

    @classmethod
    def create(cls, cfg):
        # type: (RasaNLUModelConfig) -> JiebaTokenizer
        #模块的算法加载。
        import xmnlp
        #xmnlp.set_stopword('/path/to/stopword.txt') # 用户自定义停用词
        #下次补上具体creat的代码
        doc = """这件衣服的质量也太差了吧！"""
        print('Load sentiment analyzer.Text: ', doc,'score',xmnlp.sentiment(doc))
        component_conf = cfg.for_component(cls.name, cls.defaults)
        analyzer = xmnlp.sentiment
        return cls(component_conf, analyzer)        
