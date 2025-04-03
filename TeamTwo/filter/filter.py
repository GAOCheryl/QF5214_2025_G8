import pandas as pd
import re


# Define a function to clean the text
def clean_text(text):
    text = text.lower()  # to lowercase
    text = re.sub(r'\s{2,}.*', '', text) # Remove the ticker symbol of the non-body part (after two spaces)
    text = re.sub(r'[^\x00-\x7F]+', '', text) # to remove emojis
    text = re.sub(r'http\S+|www.\S+', '', text)  # Remove URL
    text = re.sub(r'@\w+', '', text)  # Remove @UserID
    text = re.sub(r'#[A-Za-z0-9]+', '', text)  # to remove #Tag
    text = re.sub(r'RT\s+', '', text)  # Remove the RT forwarding tag
    text = re.sub(r'[^\w\s$]', '', text) # Remove special characters and punctuation marks (keep the $ sign as it is used to represent ticker symbols)
    text = ' '.join(text.split()) # Remove extra spaces
    return text


def filter_irrelevant_comments(text, target_stock):
    """
    Excludes text that does not contain the target stock (e.g., $ABDE) and contains other stocks ($XXXX).
    return: A list of eligible reviews
    """
    # Matches all ticker symbols of the form $xxxx
    stock_symbols = re.findall(r'\$\w+', text)
    
    # If there is no ticker symbol, return FALSE directly
    if not stock_symbols:
        return False
    
    # Filter out $abde
    filtered_symbols = [symbol for symbol in stock_symbols if symbol.lower() != f'${target_stock.lower()}']
    
    # Condition: The text cannot contain "abde", but it does contain other ticker symbols
    return target_stock.lower() not in text.lower() and len(filtered_symbols) > 0


def remove_stock_symbols_flexible(text):
    """
    Delete all forms such as $XXX... (3-5 letters, case-insensitive).
    """
    pattern = re.compile(r'\$\w+', re.IGNORECASE)
    cleaned_text = pattern.sub('', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text


def filter_text(text):
    """
    Filter out ad text, short text, and nonsensical text
    Returning True retains the text, and False indicates filtering out
    """
    cleaned = text
   # 1. Filter empty text or short text (less than 5 meaningful words)
    words = [w for w in cleaned.split() if len(w) > 1]  #  Ignore single-letter words
    if len(words) < 5:
        return False
    
    # 2. Filter nonsensical text (check if it contains enough real words)
    stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'of', 'to', 'in', 'it', 'this', 'that', 'i'}
    content_words = [w for w in words if w not in stop_words]
    if len(content_words) < 3:  
        return False
    
    # 3. Filter ad text (using keywords and pattern matching)
    ad_patterns = [
        r'\b(?:live\s+support|trade\s+ideas|scanner|analysis|market)\b',
        r'\b(?:join\s+now|subscribe|limited\s+time|offer|discount)\b',
        r'\b(?:don\'?t\s+(?:miss|lose)|money\s+back)\b',
        r'\b(?:day\s+trading|stock\s+market|profit|earn\s+money)\b',
        r'\b(?:click|link|website|visit|check\s+out)\b',
        r'\b(?:free\s+trial|bonus|promo|giveaway)\b',
        r'\!{2,}|\?{2,}', 
        r'\b(?:guarantee|results|performance|success)\b'
    ]
    
    for pattern in ad_patterns:
        if re.search(pattern, cleaned):
            return False
    
    # 4. Filter non-substantive text (check if it's just repetitive characters or words)
    if len(set(words)) < 2:  # All words are the same
        return False
    
    # 5. Filter pure symbols or numerical text
    if re.fullmatch(r'[\d\W_]+', cleaned.replace(' ', '')):
        return False
    
    return True




###############################################################################
# Example usage
###############################################################################

df = pd.read_csv('tweets_nasdaq100_1.csv')

# Apply the cleaning function to the Text column
df['Cleaned_Text'] = df['Text'].apply(clean_text)

# Filter DataFrames
df_related = df[df.apply(lambda row: filter_irrelevant_comments(row['Cleaned_Text'], row['Company'])==False, axis=1)].copy()

# Delete all stock like $XXX... (3-5 letters, case-insensitive).
df_related['Cleaned_Text'] = df_related['Cleaned_Text'].apply(remove_stock_symbols_flexible)

# Filter DataFrames
df_filtered = df_related[df_related['Cleaned_Text'].apply(filter_text)].copy()


# Save the cleaned data to a new Excel file
df_filtered.to_csv('filtered_data.csv', index=True)










