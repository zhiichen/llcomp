


from mako.template import Template as MakoTemplate



class TemplateParser(MakoTemplate):
   def __init__(self, *args, **kwargs):
      super(TemplateParser, self).__init__(*args, **kwargs)

