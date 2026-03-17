import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy_financial as npf
from scipy.optimize import root_scalar

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trust Housing Deployment",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── minimal custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* tighter sidebar */
    [data-testid="stSidebar"] { min-width: 310px; max-width: 310px; }
    /* metric card row */
    div[data-testid="metric-container"] {
        background: #f8f9fb;
        border: 1px solid #e2e6ea;
        border-radius: 8px;
        padding: 12px 16px;
    }
    /* section headers in sidebar */
    .sidebar-section {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #6c757d;
        margin: 16px 0 6px 0;
    }
    /* headline callout */
    .grant-callout {
        background: #eef6ff;
        border-left: 4px solid #3b82f6;
        border-radius: 4px;
        padding: 10px 14px;
        font-size: 0.9rem;
        color: #1e3a5f;
        margin-bottom: 16px;
    }
    /* keep sidebar expanded */
    [data-testid="stSidebarCollapseButton"] {
    display: none;
    }

</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# STUB — replace with:  import beepy as bep
# ══════════════════════════════════════════════════════════════════════════════
def simulate_mv_returns(dict_inputs):
    # long multi-variate simulation based on available asset class granularity
    log_mu = dict_inputs['df_returns'].iloc[:,0].values + dict_inputs['inflation']
    log_vols = dict_inputs['df_vols'].iloc[:,0].values
    log_corr = dict_inputs['df_corr'].values
    it = dict_inputs['iterations'] * dict_inputs['years']

    s = np.asarray((np.log(1+(log_vols/(1+log_mu))**2))**.5)
    m = np.asarray(np.log(1+log_mu) - s**2/2)

    return np.exp(np.random.multivariate_normal(m, (log_corr*s).T*s, size=it))-1
    
def find_house_price(dict_inputs):
    try:
        sol = root_scalar(
            lambda hp: housing_cost_solve(hp, dict_inputs),
            bracket=[1, 10_000_000],
            method='brentq'
        )
        return sol.root if sol.converged else np.nan
    except ValueError:
        return np.nan

def find_input_value(dict_inputs,key,bracket,house_price=None,method="brentq"):
    dict_inputs = dict_inputs.copy()  # avoid mutating caller state

    def f(x):
        dict_inputs[key] = x
        return housing_cost_solve(house_price, dict_inputs)

    sol = root_scalar(f, bracket=bracket, method=method)
    return sol.root if sol.converged else np.nan

def initialize_sim(dict_inputs):
    # Initiate simulation, pricipally load the simulation, configure some settings
    # Needs to be recalled if iterations, years, portfolios, beginning year, asset class assumptions change
    dict_inputs['df_cov'] = (dict_inputs['df_corr']*dict_inputs['df_vols'].iloc[:,0].values).T*dict_inputs['df_vols'].iloc[:,0].values
    dict_inputs['end_yr'] = dict_inputs['curr_sim_yr'] + dict_inputs['years'] + 1
    dict_inputs['mv_simulation'] = simulate_mv_returns(dict_inputs)
    dict_inputs['port_simulation'] = np.dot(dict_inputs['mv_simulation'],dict_inputs['df_portfolios'].values)
    dict_inputs['arr_deflator'] = (1+dict_inputs['inflation'])**np.arange(dict_inputs['years']+1)
    return dict_inputs

def housing_profile(dict_inputs,fed_inc_tax_rate = .22,house_price_apprec = .04,inflation = .03):
    loan_value = dict_inputs['house_price'] - dict_inputs['down_pmt'] - dict_inputs['grant']
    tax_deductible = ['Interest','Property Tax','PMI']
    df_housing_cost = pd.DataFrame([],index=range(1,dict_inputs['mtg_term']+1),columns=['Home Value','Principal','Interest','Loan Balance (beg)','Equity $','Equity %','PMI','Property Tax','Insurance','Monthly CF','Deductible','Tax Credit','CF ex Princ/Tax Credits'])
    df_housing_cost['Home Value'] = (dict_inputs['house_price']*(1+house_price_apprec)**((df_housing_cost.index-1)/12)).values
    if dict_inputs['interest_only']:
        df_housing_cost['Principal'] = 0
        df_housing_cost['Interest'] = dict_inputs['mtg_rate'] / 100 / 12 * loan_value
    else:    
        df_housing_cost['Principal'] = npf.ppmt(dict_inputs['mtg_rate']/100/12,range(1,dict_inputs['mtg_term']+1),dict_inputs['mtg_term'],-loan_value)
        df_housing_cost['Interest'] = npf.ipmt(dict_inputs['mtg_rate']/100/12,range(1,dict_inputs['mtg_term']+1),dict_inputs['mtg_term'],-loan_value)
    df_housing_cost['Loan Balance (beg)'] = loan_value - pd.concat([pd.Series([0]),df_housing_cost['Principal'].cumsum().iloc[0:-1]]).values
    df_housing_cost['Equity $'] = df_housing_cost['Home Value'] - df_housing_cost['Loan Balance (beg)']
    df_housing_cost['Equity %'] = df_housing_cost['Equity $'] / df_housing_cost['Home Value'] * 100
    # need to adjust pmi to hold constant and then step off at 22% equity
    # PMI calculated at day 1 and carried until the borrower has reappraisal to confirm 20% equity; note commented out original had sliding pmi scale
    # df_housing_cost['PMI'] = (df_housing_cost['Equity %']/100).apply(pmi_rate) * df_housing_cost['Loan Balance (beg)'] / 12 * (1 - dict_inputs['private_loan'])
    df_housing_cost['PMI'] = pmi_rate(df_housing_cost['Equity %'].iloc[0]/100) * df_housing_cost['Loan Balance (beg)'].iloc[0] / 12
    df_housing_cost.loc[df_housing_cost.index[df_housing_cost['Equity %'] > 20],'PMI'] = 0 # zero out PMI once equity at least 20% (borrower has to enquire)
    if dict_inputs.get('private_loan', False):
        df_housing_cost['PMI'] = 0
    df_housing_cost['Property Tax'] = df_housing_cost['Home Value'] * dict_inputs['property_tax_rate'] / 100 / 12
    df_housing_cost['Insurance'] = df_housing_cost['Home Value'] * dict_inputs['insurance_rate'] / 100 / 12
    df_housing_cost['Monthly CF'] = df_housing_cost[['Principal','Interest','PMI','Property Tax','Insurance']].sum(axis=1)
    df_housing_cost['Deductible'] = df_housing_cost[['Interest','PMI','Property Tax']].sum(axis=1)
    df_housing_cost['Interest-Credit'] = df_housing_cost['Interest'] * fed_inc_tax_rate
    df_housing_cost['PMI-Credit'] = df_housing_cost['PMI'] * fed_inc_tax_rate
    df_housing_cost['Property Tax-Credit'] = df_housing_cost['Property Tax'] * fed_inc_tax_rate
    df_housing_cost['Tax Credit'] = df_housing_cost['Deductible'] * fed_inc_tax_rate
    df_housing_cost['CF ex Princ/Tax Credits'] = df_housing_cost['Monthly CF'] - df_housing_cost['Principal'] - df_housing_cost['Tax Credit']
    return df_housing_cost

def housing_cost_solve(house_price, dict_inputs):
    loan_value = house_price - (dict_inputs['grant'] + dict_inputs['down_pmt'])
    equity_pct = 1 - loan_value / house_price

    interest_only = dict_inputs.get("interest_only", False)
    pmi_mult = pmi_multiplier(dict_inputs)

    if interest_only:
        monthly_mtg = dict_inputs['mtg_rate'] / 100 / 12 * loan_value
    else:
        monthly_mtg = npf.pmt(rate=dict_inputs['mtg_rate'] / 100 / 12, nper=dict_inputs['mtg_term'], pv=-loan_value)

    monthly_other = (pmi_mult * pmi_rate(equity_pct) * loan_value + (
                dict_inputs['property_tax_rate'] / 100 + dict_inputs['insurance_rate'] / 100) * house_price) / 12

    max_monthly = dict_inputs['income'] * dict_inputs['debt_to_income'] / 12

    return monthly_mtg + monthly_other - max_monthly

def pmi_rate(equity):
    return max(0, -0.04 * equity + 0.008)

def pmi_multiplier(dict_inputs):
    return 0 if dict_inputs.get("private_loan", False) else 1
# ══════════════════════════════════════════════════════════════════════════════
# CASH-FLOW BUILDER  (replaces graph_housing_input)
# ══════════════════════════════════════════════════════════════════════════════
def build_housing_cfs(h):
    term = h['house_term']
    cap  = h['house_cap'] * h['house_usage_rate']
    rate = h['int_rate']
    tax  = h['tax_on_interest']
    is_io = h.get('interest_only', True)

    arr_principal = np.zeros(term + 1)
    arr_interest  = np.zeros(term + 1)
    arr_net_cf    = np.zeros(term + 1)

    if h['is_grant']:
        arr_net_cf[0] = h['grant_cap']
    else:
        arr_net_cf[0] = cap
        periods = np.arange(1, term + 1)
        if is_io:
            arr_interest[1:]   = rate * cap
            arr_principal[-1]  = cap
        else:
            arr_interest[1:]   = npf.ipmt(rate, periods, term, -cap)
            arr_principal[1:]  = npf.ppmt(rate, periods, term, -cap)
        arr_net_cf[1:] = -(arr_principal[1:] + arr_interest[1:] * (1 - tax))

    return arr_net_cf, arr_interest, arr_principal


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION SETUP  (cached — only reruns when sim-level inputs change)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def run_simulation(iterations, years, inflation, portfolios_tuple):
    df_portfolios = pd.DataFrame(
        [[.5,.6,.7,.8,.9,1],[.5,.4,.3,.2,.1,0]],
        index=['Equity','Munis'],
        columns=['50/50','60/40','70/30','80/20','90/10','100/0']
    )
    df_returns = pd.DataFrame([.05,.015], index=['Equity','Munis'], columns=['Real Arithmetic Return'])
    df_vols    = pd.DataFrame([.17,.07],  index=['Equity','Munis'], columns=['Standard Deviation'])
    df_corr    = pd.DataFrame([[1,.25],[.25,1]], index=['Equity','Munis'], columns=['Equity','Munis'])

    d = {
        'iterations': iterations, 'years': years, 'inflation': inflation,
        'curr_sim_yr': 2025,
        'df_portfolios': df_portfolios, 'df_returns': df_returns,
        'df_vols': df_vols, 'df_corr': df_corr,
    }
    d['df_cov']    = (d['df_corr'] * d['df_vols'].iloc[:,0].values).T * d['df_vols'].iloc[:,0].values
    d['end_yr']    = d['curr_sim_yr'] + d['years'] + 1
    d              = initialize_sim(d)
    return d


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("Inputs")
    
    # ── 0. Financing Conventions ───────────────────────────────────────────────
    with st.expander("Financing Conventions", expanded=False):
        base_rate    = st.number_input("Base Mortgage Rate (%)",   value=6.0,     step=0.25, format="%.2f")
        dti_pct      = st.number_input("Housing % of Gross Income", value=28, step=1, min_value=1, max_value=50,help="Front End Ratio: Max % of gross monthly income allocated to housing costs (PITI): principal, interest, PMI, property taxes, insurance and HOAs (not incl here)")
        dti          = dti_pct / 100
        prop_tax     = st.number_input("Property Tax Rate (%)",    value=1.2,     step=0.1,  format="%.2f",help="Annual Cost of Property Tax, % of Value, Impacts Financing Capacity Under the Front End Ratio")
        insurance    = st.number_input("Insurance Rate (%)",       value=0.5,     step=0.1,  format="%.2f",help="Annual Cost of Insurance, % of Value, Impacts Financing Capacity Under the Front End Ratio")
        mtg_term_yrs = st.selectbox("Mortgage Term (years)",       [30, 20, 15], index=0)
        mtg_term     = mtg_term_yrs * 12
        inflation    = st.slider("Inflation (%)",                  1, 6, 3) / 100
        
    # ── 1. Beneficiary Finances ───────────────────────────────────────────────
    with st.expander("Borrower Assumptions", expanded=False):
        income_k     = st.number_input("Annual Income ($000s)",        value=200, step=10)
        income       = income_k * 1_000
        down_pmt_k   = st.number_input("Down Payment ($000s)",         value=25, step=5)
        down_pmt     = down_pmt_k * 1_000

    # ── 2. Support Terms ─────────────────────────────────────────────────────
    with st.expander("Support Design", expanded=False):
        loan_rate    = st.number_input("Subsidized Loan Rate (%)", value=4.5,  step=0.25, format="%.2f")
        interest_only = st.checkbox("Interest-Only Loan",          value=True)
        private_loan = st.checkbox("Waive PMI (private/subsidized loan)", value=True,help="Private or subsidized loans typically waive PMI requirement")
        horizon      = st.slider("Show Home Equity as of Year:",   3, 30, 10)
        fed_tax_rate_int = st.slider("Federal Income Tax Rate (%)",    10, 40, 22,help="Married Filing Jointly Rates: 22% up to \$207k/24% to \$395k/32% to \$500k/ 35% to \$750k/37% Above")
        fed_tax_rate = fed_tax_rate_int / 100

    # ── 3. Trust Assumptions ──────────────────────────────────────────────────
    with st.expander("Trust Assumptions", expanded=False):
        portfolio    = st.selectbox("Portfolio Mix",
                                    ['50/50','60/40','70/30','80/20','90/10','100/0'],
                                    index=2)
        tax_interest = st.slider("Tax on Interest (%)", 20, 50, 40) / 100
        st.markdown('<div class="sidebar-section">Admin Costs — Loan</div>', unsafe_allow_html=True)
        loan_txn_init= st.number_input("Upfront ($000s)",    value=50, step=1_000, key='loan_txn')
        loan_txn     = loan_txn_init * 1000
        loan_annual_init  = st.number_input("Annual ($000s)",     value=1,    step=100,   key='loan_annual')
        loan_annual = loan_annual_init * 1000
        st.markdown('<div class="sidebar-section">Admin Costs — Grant</div>', unsafe_allow_html=True)
        grant_txn_init    = st.number_input("Upfront ($000s)",    value=25, step=1_000, key='grant_txn')
        grant_txn = grant_txn_init * 1000


# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE
# ══════════════════════════════════════════════════════════════════════════════

# ── Base inputs dict ──────────────────────────────────────────────────────────
base = {
    'down_pmt': down_pmt, 'mtg_rate': base_rate, 'grant': 0,
    'interest_only': False, 'private_loan': False,
    'property_tax_rate': prop_tax, 'insurance_rate': insurance,
    'debt_to_income': dti, 'income': income, 'mtg_term': mtg_term,
}
base['house_price'] = find_house_price(base)

# ── Loan scenario ─────────────────────────────────────────────────────────────
loan_d = base.copy()
loan_d.update({'mtg_rate': loan_rate, 'interest_only': interest_only, 'private_loan': private_loan})
loan_d['house_price'] = find_house_price(loan_d)

# ── Grant scenario — same house price, solve for grant ───────────────────────
grant_d = base.copy()
grant_d['house_price'] = loan_d['house_price']
grant_d['grant'] = find_input_value(
    grant_d, 'grant', [0, 1_000_000],
    house_price=grant_d['house_price'], method="brentq"
)

# ── Scenario container ────────────────────────────────────────────────────────
scenarios = {
    'Unsupported': base.copy(),
    'Loan':        loan_d,
    'Grant':       grant_d,
}

dict_outputs = {k: housing_profile(v, fed_inc_tax_rate=fed_tax_rate, inflation=inflation)
                for k, v in scenarios.items()}

# ── Derived headline numbers ──────────────────────────────────────────────────
grant_amt          = grant_d['grant']
loan_capital       = loan_d['house_price'] - loan_d['down_pmt']
surplus_reinvested = loan_capital - grant_amt

# ── Trust simulation ──────────────────────────────────────────────────────────
sim = run_simulation(10_000, 100, inflation, tuple(portfolio))
arr_returns = (
    sim['port_simulation']
    .reshape(sim['years'], sim['iterations'], sim['df_portfolios'].shape[1])
)[:, :, sim['df_portfolios'].columns.get_loc(portfolio)]

dict_housing_loan = {
    'house_cap': loan_capital, 'house_usage_rate': 1,
    'house_purchase_age': 28, 'int_rate': loan_rate / 100,
    'tax_on_interest': tax_interest, 'house_term': mtg_term_yrs,
    'is_grant': False, 'grant_cap': 0,
    'interest_only': interest_only,
}
dict_housing_grant = {
    'house_cap': 0, 'house_usage_rate': 1,
    'house_purchase_age': 28, 'int_rate': loan_rate / 100,
    'tax_on_interest': tax_interest, 'house_term': mtg_term_yrs,
    'is_grant': True, 'grant_cap': grant_amt,
}

arr_loan_cf,  arr_int_loan,  arr_princ_loan  = build_housing_cfs(dict_housing_loan)
arr_grant_cf, arr_int_grant, arr_princ_grant = build_housing_cfs(dict_housing_grant)

arr_mv_loan  = np.zeros([mtg_term_yrs + 1, 10_000])
arr_mv_grant = np.zeros([mtg_term_yrs + 1, 10_000])
arr_mv_grant[0, :] = (arr_loan_cf[0] + loan_txn) - (arr_grant_cf[0] + grant_txn)

for yr in range(mtg_term_yrs):
    arr_mv_loan[yr+1, :] = (arr_mv_loan[yr] * (1 + arr_returns[yr, :])
                             - arr_loan_cf[yr+1]
                             - loan_annual)
    arr_mv_grant[yr+1, :] = (arr_mv_grant[yr] * (1 + arr_returns[yr, :])
                              - arr_grant_cf[yr+1])

# breakeven return (year mtg_term_yrs)
breakeven_pct = (
    ((arr_mv_loan.mean(axis=1)[:-1] - arr_loan_cf[-1]) / arr_mv_grant[0, 0])
    ** (1 / np.arange(1, mtg_term_yrs + 1)) - 1
) * 100
breakeven_final = breakeven_pct[-1]

avg_port_return = (np.exp(np.log1p(arr_returns).mean()) - 1) * 100


# ══════════════════════════════════════════════════════════════════════════════
# PALETTE
# ══════════════════════════════════════════════════════════════════════════════
palette   = px.colors.qualitative.Prism
color_map = {k: palette[i % len(palette)] for i, k in enumerate(scenarios)}


# ══════════════════════════════════════════════════════════════════════════════
# HEADER + HEADLINE METRICS
# ══════════════════════════════════════════════════════════════════════════════
st.title("Housing Deployment: Loan vs. Grant")
st.markdown(
    f'<div class="grant-callout">'
    f'To match the purchasing power of a <b>{loan_rate}% subsidized loan</b>, '
    f'the trust would need to deploy a grant of '
    f'<b>${grant_amt:,.0f}</b> — '
    f'${surplus_reinvested:,.0f} less capital than the loan.'
    f'</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Loan Capital Deployed",  f"${loan_capital:,.0f}")
c2.metric("Grant Required",         f"${grant_amt:,.0f}")
c3.metric("Surplus (Loan − Grant)", f"${surplus_reinvested:,.0f}")
c4.metric(f"Breakeven Return (Yr {mtg_term_yrs})",
          f"{breakeven_final:.1f}%",
          delta=f"{breakeven_final - avg_port_return:.1f}% vs portfolio avg",
          delta_color="inverse")

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["📋  Beneficiary Impact", "🏦  Trust Opportunity Cost"])

# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    fig1 = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Initial Monthly Housing Cost',
            f'Initial Monthly Net Cost (Principal & Tax Deductibility) — Year 1',
            f'Home Equity — Year {horizon} (4% Price Appreciation)',
            'Inflation Adjusted Annual Net Cost',
        ],
        vertical_spacing=0.12,
    )

    for i, (key, data) in enumerate(dict_outputs.items()):
        hp = int(data.loc[1, 'Home Value'].round(-3) / 1000)
        legend_name = (f"{key} — Home Price ${hp}k" if i == 0 else f"{key} — ${hp}k")
        color = color_map[key]

        # col 1: stacked monthly cost components
        opacity = 1.0
        for comp in ['Principal','Interest','Property Tax','Insurance','PMI']:
            fig1.add_trace(go.Bar(
                name=key, x=[key], y=[data.loc[1, comp]],
                text=[comp], textfont=dict(size=8),
                showlegend=False,
                marker=dict(color=color, opacity=opacity)
            ), row=1, col=1)
            opacity -= 0.2

        # col 2: net cost bar + principal + tax credit
        fig1.add_trace(go.Bar(
            name=legend_name, x=[key],
            y=data.loc[[1], 'CF ex Princ/Tax Credits'],
            text=data.loc[[1], 'CF ex Princ/Tax Credits'].round(0),
            textfont=dict(size=11, color='black', family="Arial Black"),
            texttemplate="$%{y:,.0f}",
            marker=dict(color=color, line=dict(color="black", width=2)),
            legendrank=3 - i,
        ), row=1, col=2)
        fig1.add_trace(go.Bar(
            name=key, x=[key], y=data.loc[[1], 'Principal'],
            text=['Principal'], textfont=dict(size=8),
            showlegend=False, marker=dict(color=color, opacity=0.4)
        ), row=1, col=2)
        fig1.add_trace(go.Bar(
            name=key, x=[key], y=data.loc[[1], 'Tax Credit'],
            text=['Tax Credit'], textfont=dict(size=8),
            showlegend=False, marker=dict(color=color, opacity=0.25)
        ), row=1, col=2)

        # col 3: equity waterfall
        apprec      = data.loc[horizon*12+1, 'Home Value'] - data.loc[1, 'Home Value']
        paid_princ  = data.loc[1:horizon*12, 'Principal'].sum()
        init_equity = data.loc[1, 'Equity $']
        total_eq    = init_equity + paid_princ + apprec

        for y_val, label, op in [
            (apprec,     'Appreciation', 0.75),
            (paid_princ, 'Princ Paid',   0.50),
            (init_equity,'Down Pmt',     1.00),
        ]:
            fig1.add_trace(go.Bar(
                name=key, x=[key], y=[y_val],
                text=[label], textfont=dict(size=8),
                showlegend=False, marker=dict(color=color, opacity=op)
            ), row=2, col=1)

        fig1.add_trace(go.Scatter(
            name=key, x=[key], y=[total_eq * 1.09],
            text=[f"${total_eq:,.0f}"],
            textfont=dict(size=10), mode='text',
            textposition='top center', showlegend=False,
        ), row=2, col=1)

        # col 4: inflation-adjusted annual net cost
        ann_cf = (
            data['CF ex Princ/Tax Credits']
            .groupby(np.arange(len(data)) // 12).sum()
            / (1 + inflation) ** pd.Series(range(1, 31))
        )
        fig1.add_trace(go.Scatter(
            name=legend_name, x=list(range(1, 31)), y=ann_cf,
            mode='lines', line=dict(color=color), showlegend=False,
        ), row=2, col=2)

    notes = (
        f"Income ${int(base['income']/1000)}k | "
        f"Borrower Income Tax Bracket {int(fed_tax_rate*100)}% | "
        f"Rates: " + " / ".join(f"{k} {scenarios[k]['mtg_rate']}%" for k in scenarios) +
        f" | Inflation {int(100*inflation)}%"
    )
    fig1.add_annotation(
        text=notes, xref="x3 domain", yref="y4 domain",
        x=0, y=-0.2, showarrow=False,
        font=dict(size=10, color="gray"), align="left",
    )
    fig1.update_layout(
        height=700, barmode="stack",
        showlegend=False,
        # legend=dict(x=0.5, y=1.08, orientation='h', xanchor="center"),
        title=f"Purchasing Power: Loan vs Grant — Beneficiary View",
        margin=dict(t=100),
    )
    fig1.update_annotations(font=dict(size=12))
    fig1.update_yaxes(tickformat="$,.0f", rangemode="tozero", row=1, col=1)
    fig1.update_yaxes(tickformat="$,.0f", rangemode="tozero", row=1, col=2)
    fig1.update_yaxes(tickformat="$,.0f", rangemode="tozero", row=2, col=1)
    fig1.update_xaxes(title_text="Year", row=2, col=2)

    st.plotly_chart(fig1, use_container_width=True)


# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    fig2 = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Trust Cash Flows (000s)',
            'Required Return for Grant to Match Loan',
            'Probability Grant Surplus Exceeds Loan',
            'Invested Grant Less Loan Value (000s)',
        ],
        vertical_spacing=0.12, shared_xaxes=True,
    )

    # Cash flows — stacked principal / interest-net bars for loan
    fig2.add_trace(go.Bar(
        name='Loan — Principal',
        y=arr_princ_loan / 1000,
        marker=dict(color='#f97316'), showlegend=True,
    ), row=1, col=1)
    fig2.add_trace(go.Bar(
        name='Loan — Interest (after tax)',
        y=(arr_int_loan * (1 - tax_interest)) / 1000,
        marker=dict(color='#fb923c'), showlegend=True,
    ), row=1, col=1)
    fig2.add_trace(go.Bar(
        name='Grant',
        x=[0], y=[arr_grant_cf[0] / 1000],
        marker=dict(color='#3b82f6'), showlegend=True,
    ), row=1, col=1)
    # surplus bracket
    fig2.add_trace(go.Scatter(
        x=[0, 0],
        y=[arr_grant_cf[0] / 1000, arr_loan_cf[0] / 1000],
        mode='markers+lines+text', text=['', 'Surplus'],
        textposition='bottom right',
        marker=dict(color='black', symbol='line-ew', size=8, line=dict(width=2)),
        showlegend=False,
    ), row=1, col=1)

    # Breakeven return curve
    fig2.add_trace(go.Scatter(
        y=breakeven_pct, mode='lines',
        line=dict(color='#16a34a', width=2),
        name='Breakeven Return', showlegend=False,
    ), row=1, col=2)
    fig2.add_hline(
        row=1, col=2, y=avg_port_return,
        line=dict(color='black', width=2),
        annotation_text=f'Avg Portfolio Return ({avg_port_return:.1f}%)',
        annotation_font_size=10,
    )

    # Probability grant surplus > loan
    prob = ((arr_mv_loan[1:-1, :] - arr_loan_cf[-1]) < arr_mv_grant[1:-1, :]).mean(axis=1) * 100
    fig2.add_trace(go.Scatter(
        y=np.minimum(prob, 50), mode='lines',
        line=dict(color='#ef4444'), fill='tozeroy',
        showlegend=False,
    ), row=2, col=1)
    fig2.add_trace(go.Scatter(
        y=prob, mode='lines',
        line=dict(color='black'), fill='tonexty',
        showlegend=False,
    ), row=2, col=1)
    fig2.add_hline(row=2, col=1, y=50, line=dict(color='gray', dash='dot'))

    # Box plot: grant surplus distribution
    diff_df = pd.DataFrame(
        (arr_mv_grant[1:-1, :] - (arr_mv_loan[1:-1, :] - arr_loan_cf[-1])) / 1000,
        index=range(1, mtg_term_yrs)
    ).stack().reset_index()
    diff_df.columns = ['year', 'iter', 'value']
    fig2.add_trace(go.Box(
        x=diff_df['year'], y=diff_df['value'],
        boxpoints=False, marker=dict(color='#6b7280'),
        showlegend=False,
    ), row=2, col=2)
    fig2.add_hline(row=2, col=2, y=0, line=dict(color='black'))

    p25 = np.percentile(diff_df['value'], 25)
    p75 = np.percentile(diff_df['value'], 75)
    fig2.update_layout(
        height=650,
        barmode='stack',
        title='Loan vs Grant — Trust Opportunity Cost Analysis',
        showlegend=False,
        # legend=dict(x=0, y=1.1, orientation='h', xanchor='left'),
        yaxis2_range=[0, 20],
        yaxis4_range=[p25 * 1.1 if p25 < 0 else p25 * 0.9, p75 * 1.1],
        margin=dict(t=100),
    )
    fig2.update_yaxes(tickformat="$,.3s", row=1, col=1)
    fig2.update_yaxes(tickformat=".1f",   row=1, col=2, title_text="%")
    fig2.update_yaxes(tickformat=".0f",   row=2, col=1, title_text="% probability")
    fig2.update_yaxes(tickformat="$,.3s", row=2, col=2)
    fig2.update_xaxes(title_text="Year",  row=2, col=1)
    fig2.update_xaxes(title_text="Year",  row=2, col=2)

    st.plotly_chart(fig2, use_container_width=True)

    # admin cost context note
    st.caption(
        f"Admin costs — Loan: ${loan_txn:,.0f} upfront + ${loan_annual:,.0f}/yr | "
        f"Grant: ${grant_txn:,.0f} upfront | "
        f"Tax on interest: {int(tax_interest*100)}% | "
        f"Portfolio: {portfolio}"
    )