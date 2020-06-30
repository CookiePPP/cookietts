""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. '''
from CookieTTS.utils.text import cmudict

_pad        = '_'
_punctuation = '!\'(),.:;? '
_special = '-'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_custom ='☺␤☻♥♦♣♠•◘○◙♂♀♪♫☼►◄↕#¶§▬↨↑↓→←∟↔▲'
# ☺     = Start Token Char
# ␤     = Stop Token Char
# #     = Singing Hold Note Char

_arpabet = ['@' + s for s in cmudict.valid_symbols]

symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters) + _arpabet + list(_custom)