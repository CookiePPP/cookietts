from unidecode import unidecode

class ARPA:
    def __init__(self, dict_path):
        self.arpadict = self.load_arpadict(dict_path)
        self.punc = "!?,.;:␤#~-_'\"()[]\n"
    
    def load_arpadict(self, dict_path):
        # load dictionary as lookup table
        arpadict = {unidecode(line.split()[0]): unidecode(' '.join(line.split()[1:]).strip()) for line in open(dict_path, 'r')}
        return arpadict
    
    def get(self, text):
        """Convert block of text into ARPAbet."""
        out = []
        for word in text.split(" "):
            end_chars = ''; start_chars = ''
            while any(elem in word for elem in self.punc) and len(word) > 1:
                if word[-1] in self.punc:
                    end_chars = word[-1] + end_chars
                    word = word[:-1]
                elif word[0] in self.punc:
                    start_chars = start_chars + word[0]
                    word = word[1:]
                else:
                    break
            try:
                word = "{" + str(self.arpadict[word.upper()]) + "}"
            except KeyError:
                pass
            out.append((start_chars + (word or '') + end_chars).rstrip())
        return ' '.join(out)