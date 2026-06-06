#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arabic to IPA converter - Version 2
Fixed shadda handling: Shadda appears on the consonant, not after the vowel.
"""

import unicodedata


class ArabicToIPA:
    """Convert Arabic text to IPA with proper vowel and shadda handling."""
    
    # Consonant mapping
    CONSONANTS = {
        'ب': 'b', 'ت': 't', 'ث': 'θ', 'ج': 'd͡ʒ',
        'ح': 'ħ', 'خ': 'x', 'د': 'd', 'ذ': 'ð',
        'ر': 'r', 'ز': 'z', 'س': 's', 'ش': 'ʃ',
        'ص': 'sˤ', 'ض': 'dˤ', 'ط': 'tˤ', 'ظ': 'ðˤ',
        'ع': 'ʕ', 'غ': 'ɣ', 'ف': 'f', 'ق': 'q',
        'ك': 'k', 'ل': 'l', 'م': 'm', 'ن': 'n',
        'ه': 'h', 'ء': 'ʔ',
        'ة': 't',  # Ta marbuta
    }
    
    # Long vowels
    LONG_VOWELS = {
        'ا': 'aː',  # Alif
        'و': 'uː',  # Waw
        'ي': 'iː',  # Ya
    }
    
    # Short vowels (diacritics)
    SHORT_VOWELS = {
        'َ': 'a',   # Fatha
        'ُ': 'u',   # Damma
        'ِ': 'i',   # Kasra
    }
    
    SUKUN = 'ْ'
    SHADDA = 'ّ'
    
    TANWIN = {
        'ً': 'an',
        'ٌ': 'un',
        'ٍ': 'in',
    }
    
    SPECIAL_ALIF = {
        'أ': 'ʔa',
        'إ': 'ʔi',
        'آ': 'ʔaː',
    }
    
    OTHER = {
        'ئ': 'ʔi',
        'ؤ': 'ʔu',
        'ى': 'aː',
        'ٰ': 'aː',
    }
    
    def convert(self, text: str) -> str:
        """Convert Arabic text to IPA."""
        text = unicodedata.normalize('NFC', text)
        
        result = []
        i = 0
        chars = list(text)
        
        while i < len(chars):
            char = chars[i]
            
            if char == ' ':
                result.append(' ')
                i += 1
                continue
            
            if char in self.SPECIAL_ALIF:
                result.append(self.SPECIAL_ALIF[char])
                i += 1
                continue
            
            if char in self.OTHER:
                result.append(self.OTHER[char])
                i += 1
                continue
            
            # Consonant
            if char in self.CONSONANTS:
                consonant = self.CONSONANTS[char]
                
                # Parse the diacritics that follow
                has_shadda = False
                vowel = None
                next_idx = i + 1
                
                # Shadda can come before or after the vowel diacritic
                # Typical order: consonant + shadda + vowel OR consonant + vowel + shadda
                # We need to check both
                
                diacritics = []
                while next_idx < len(chars) and chars[next_idx] in self.SHORT_VOWELS or \
                      (next_idx < len(chars) and chars[next_idx] == self.SHADDA) or \
                      (next_idx < len(chars) and chars[next_idx] == self.SUKUN):
                    diacritics.append(chars[next_idx])
                    next_idx += 1
                
                # Process diacritics
                for d in diacritics:
                    if d == self.SHADDA:
                        has_shadda = True
                    elif d in self.SHORT_VOWELS:
                        vowel = self.SHORT_VOWELS[d]
                    elif d == self.SUKUN:
                        vowel = ''  # Explicitly no vowel
                
                # Check for matres lectionis (long vowel letter)
                if vowel and next_idx < len(chars):
                    long_char = chars[next_idx]
                    if long_char == 'ا' and vowel == 'a':
                        vowel = 'aː'
                        next_idx += 1
                    elif long_char == 'و' and vowel == 'u':
                        vowel = 'uː'
                        next_idx += 1
                    elif long_char == 'ي' and vowel == 'i':
                        vowel = 'iː'
                        next_idx += 1
                
                # Build output
                if has_shadda:
                    result.append(consonant)
                    result.append(consonant)
                else:
                    result.append(consonant)
                
                if vowel:
                    result.append(vowel)
                
                i = next_idx
                continue
            
            # Long vowel standalone
            if char in self.LONG_VOWELS:
                result.append(self.LONG_VOWELS[char])
                i += 1
                continue
            
            # Tanwin
            if char in self.TANWIN:
                result.append(self.TANWIN[char])
                i += 1
                continue
            
            # Skip shadda if it appears alone
            if char == self.SHADDA:
                i += 1
                continue
            
            # Unknown
            i += 1
        
        return ''.join(result)


def normalize_ipa(ipa: str) -> str:
    """Convert IPA to readable form."""
    return ipa.replace('aː', 'ā').replace('uː', 'ū').replace('iː', 'ī')


def test():
    """Test the converter."""
    converter = ArabicToIPA()
    
    print("Arabic to IPA Converter - Version 2")
    print("=" * 80)
    print()
    
    test_cases = [
        ('كِتَاب', 'kitāb', 'book'),
        ('مُحَمَّد', 'muħammad', 'Muhammad - with shadda on م'),
        ('سَلَام', 'salām', 'peace'),
        ('كِتَابٌ', 'kitābun', 'book with tanwin'),
        ('مُعَلِّم', 'muʕallim', 'teacher - shadda on ل'),
        ('رَسُول', 'rasūl', 'messenger'),
        ('مَدْرَسَة', 'madrasat', 'school'),
        ('قُرْآن', 'qurʔān', 'Quran'),
        ('اللُّغَة', 'alluɣa', 'the language'),
    ]
    
    print(f"{'Arabic':<20} {'Expected':<15} {'Actual':<15} {'OK':<4} {'Notes'}")
    print("-" * 80)
    
    for ar, expected, notes in test_cases:
        ipa = normalize_ipa(converter.convert(ar))
        expected_norm = normalize_ipa(expected)
        match = '✓' if ipa == expected_norm else '✗'
        print(f"{ar:<20} {expected_norm:<15} {ipa:<15} {match:<4} {notes}")
    
    print()
    
    # Debug: show character breakdown for محمد
    print("Character breakdown for 'مُحَمَّد':")
    word = 'مُحَمَّد'
    chars = list(unicodedata.normalize('NFC', word))
    for i, c in enumerate(chars):
        name = unicodedata.name(c, 'UNKNOWN')
        print(f"  [{i}] {c} - {name}")


if __name__ == '__main__':
    test()
