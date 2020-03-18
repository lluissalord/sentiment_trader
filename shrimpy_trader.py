import shrimpy

from settings import (
    shrimpy_public_key,
    shrimpy_secret_key
)

USER_NAME = 'crypto_trader'

client = shrimpy.ShrimpyApiClient(shrimpy_public_key, shrimpy_secret_key)

create_user_response = client.create_user(USER_NAME)
user_id = create_user_response['id']

link_account_response = client.link_account(
    user_id,
    exchange_name,
    exchange_public_key,
    exchange_secret_key
)

account_id = link_account_response['id']

# wait while Shrimpy collects data for the exchange account
# only required the first time linking
time.sleep(5)

balance = client.get_balance(user_id, account_id)
holdings = balance['balances']

# Could be that EUR not availabe and has to use USD
# In tutorial is 'BTC'
consolidation_symbol = 'EUR'