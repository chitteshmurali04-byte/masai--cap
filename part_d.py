
def get_order(order_id):
    if not DB_PATH.exists(): return {'error':'order not found'}
    con=sqlite3.connect(DB_PATH); con.row_factory=sqlite3.Row
    try:
        row=con.execute('SELECT order_id,status,order_amount,delivery_minutes FROM orders WHERE order_id=? LIMIT 1',(order_id,)).fetchone()
        if row is None: return {'error':'order not found'}
        return {'order_id':row['order_id'],'status':row['status'],'order_amount':row['order_amount'],'delivery_minutes':row['delivery_minutes']}
    except sqlite3.Error: return {'error':'order not found'}
    finally: con.close()

def search_policy(query): return {'results':retrieve(query,3)}
def calculate_refund(order_amount,delivery_minutes):
    return {'refund_inr':round(float(order_amount),2) if int(delivery_minutes)>45 else 0.0}
def _ledger(tool_name,order_id,refund_inr):
    record={'timestamp':datetime.now(timezone.utc).isoformat(),'tool':tool_name,'order_id':order_id,'refund_inr':round(float(refund_inr),2)}
    with LEDGER.open('a',encoding='utf-8') as f: f.write(json.dumps(record)+'\n')
    return {'refund_inr':record['refund_inr']}
def issue_refund(order_id,refund_inr): return _ledger('issue_refund',order_id,refund_inr)
def escalate_to_human(order_id,refund_inr): return _ledger('escalate_to_human',order_id,refund_inr)
TOOL_MAP={'get_order':get_order,'search_policy':search_policy,'calculate_refund':calculate_refund,'issue_refund':issue_refund,'escalate_to_human':escalate_to_human}

# Tasks 6-8: ReAct loop, token/time/iteration guards, atomic checkpoints and resume.
class SimulatedCrash(RuntimeError): pass

def _get(obj,key,default=None): return obj.get(key,default) if isinstance(obj,dict) else getattr(obj,key,default)
def _call_parts(call):
    fn=_get(call,'function',call); name=_get(fn,'name',_get(call,'name','')); args=_get(fn,'arguments',_get(call,'arguments',{}))
    if isinstance(args,str):
        try: args=json.loads(args)
        except json.JSONDecodeError: args={}
    return str(name),dict(args or {}),str(_get(call,'id','call_'+uuid.uuid4().hex[:8]))
def _tool_calls(resp): return list(_get(resp.choices[0].message,'tool_calls',None) or [])
def _tokens(resp): return int(_get(_get(resp,'usage',{}),'total_tokens',0) or 0)
def _status(text): return text.split(':',1)[0].strip() if ':' in text else 'COMPLETED'
def _state_path(run_id): return STATE/(re.sub(r'[^A-Za-z0-9_.-]','_',run_id)+'.json')
def _save(state):
    p=_state_path(state['run_id']); tmp=p.with_suffix('.tmp'); tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2,default=str),encoding='utf-8'); tmp.replace(p)
def _load(run_id):
    p=_state_path(run_id)
    if not p.exists(): raise FileNotFoundError(f'Checkpoint not found: {run_id}')
    return json.loads(p.read_text(encoding='utf-8'))
def _fallback(s,status,reason):
    return {'status':status,'final_answer':f'{status}: {reason} Please hand this request to a human support agent.',
            'steps':s['steps'],'llm_calls':s['llm_calls'],'tokens_used':s['tokens_used'],'run_id':s['run_id'],'tools_executed':s['tools_executed']}

def _continue(s,max_iterations,token_limit,time_limit_seconds,crash_after_tools=None):
    started=time.monotonic()
    while s['llm_calls']<max_iterations:
        if time.monotonic()-started >= time_limit_seconds:
            result=_fallback(s,'FALLBACK_TIMEOUT','The agent reached its time limit.'); s['last_result']=result; _save(s); return result
        resp=mock_llm.chat(s['messages'],tools=TOOLS_SPEC); s['llm_calls']+=1; s['tokens_used']+=_tokens(resp)
        if s['tokens_used']>token_limit:
            result=_fallback(s,'FALLBACK_TOKEN_LIMIT','The agent exceeded its token budget.'); s['last_result']=result; _save(s); return result
        calls=_tool_calls(resp)
        if not calls:
            final=_content(resp); result={'status':_status(final),'final_answer':final,'steps':s['steps'],'llm_calls':s['llm_calls'],
                'tokens_used':s['tokens_used'],'run_id':s['run_id'],'tools_executed':s['tools_executed']}
            s['last_result']=result; _save(s); return result
        assistant_calls=[]
        for call in calls:
            name,args,cid=_call_parts(call); assistant_calls.append({'id':cid,'type':'function','function':{'name':name,'arguments':json.dumps(args,ensure_ascii=False)}})
        s['messages'].append({'role':'assistant','content':None,'tool_calls':assistant_calls})
        for call in calls:
            name,args,cid=_call_parts(call)
            observation=TOOL_MAP[name](**args) if name in TOOL_MAP else {'error':f'unknown tool: {name}'}
            s['steps'].append({'tool':name,'arguments':args,'observation':observation}); s['tools_executed'].append(name)
            s['messages'].append({'role':'tool','tool_call_id':cid,'name':name,'content':json.dumps(observation,ensure_ascii=False,default=str)})
            s['step_number']+=1; _save(s)  # checkpoint before any crash can happen
            if crash_after_tools is not None and len(s['tools_executed'])==crash_after_tools:
                raise SimulatedCrash(f'Crash after tool {crash_after_tools}; checkpoint saved')
    result=_fallback(s,'FALLBACK_MAX_ITERATIONS','The agent reached the maximum number of model-call iterations.')
    s['last_result']=result; _save(s); return result

def run_agent(user_message,scenario_id,max_iterations=6,token_limit=6000,time_limit_seconds=10,_run_id=None,_crash_after_tools=None):
    run_id=_run_id or f'{scenario_id}_{uuid.uuid4().hex[:10]}'
    state={'run_id':run_id,'scenario_id':scenario_id,'messages':[
        {'role':'system','content':('AGENT. Use tools to inspect orders and policies before deciding. Only completed orders can be refunded; '
        'delivery over 45 minutes qualifies for 100 percent; refunds above INR 1000 must be escalated, not issued.')},
        {'role':'user','content':user_message}], 'tokens_used':0,'llm_calls':0,'step_number':0,'steps':[],'tools_executed':[]}
    _save(state)
    return _continue(state,max_iterations,token_limit,time_limit_seconds,_crash_after_tools)

def resume_agent(run_id,max_iterations=6,token_limit=6000,time_limit_seconds=10):
    state=_load(run_id); state.pop('last_result',None)
    return _continue(state,max_iterations,token_limit,time_limit_seconds)

def _append(path,obj):
    with path.open('a',encoding='utf-8') as f: f.write(json.dumps(obj,ensure_ascii=False,default=str)+'\n')
def _log_run(scenario_id,result):
    _append(RUN_LOG,{'scenario_id':scenario_id,'status':result['status'],'final_answer':result['final_answer'],
        'llm_calls':result['llm_calls'],'tokens_used':result['tokens_used'],'steps':result['steps']})
def _msg(row): return row.get('message') or row.get('user_message') or ''
def _amount(result):
    m=re.search(r'INR\s*([0-9]+(?:\.[0-9]+)?)',result.get('final_answer',''))
    if m: return float(m.group(1))
    for step in reversed(result.get('steps',[])):
        if 'refund_inr' in step.get('observation',{}): return float(step['observation']['refund_inr'])
    return None

def run_scenarios():
    rows=read_rows(RAW_DIR/'agent_scenarios.csv'); by_id={r['scenario_id']:r for r in rows}; table=[]
    for row in rows:
        mock_llm.reset(); sid=row['scenario_id']; result=run_agent(_msg(row),sid); _log_run(sid,result)
        table.append({'scenario_id':sid,'status':result['status'],'refund_inr':_amount(result),'llm_calls':result['llm_calls'],'tokens_used':result['tokens_used']})
    return table,by_id

def run_demos(scenarios):
    s1=_msg(scenarios['S1']); mock_llm.reset(); tok=run_agent(s1,'S1_TOKEN_LIMIT_DEMO',token_limit=200); _log_run('S1_TOKEN_LIMIT_DEMO',tok)
    mock_llm.reset(); tout=run_agent(s1,'S1_TIMEOUT_DEMO',time_limit_seconds=0); _log_run('S1_TIMEOUT_DEMO',tout)
    return {'token_limit_demo_status':tok['status'],'token_limit_demo_tokens':tok['tokens_used'],
            'timeout_demo_status':tout['status'],'timeout_demo_llm_calls':tout['llm_calls']}

def run_resume_demo(scenarios):
    run_id='S1_CRASH_RESUME_DEMO'; mock_llm.reset()
    try: run_agent(_msg(scenarios['S1']),'S1_CRASH_RESUME_DEMO',_run_id=run_id,_crash_after_tools=2)
    except SimulatedCrash: pass
    result=resume_agent(run_id); _log_run('S1_CRASH_RESUME_DEMO',result)
    return {'status':result['status'],'refund_inr':_amount(result),'tools_executed_in_order':result['tools_executed'],
            'tool_execution_count':len(result['tools_executed']),'llm_calls':result['llm_calls'],'tokens_used':result['tokens_used']}

def main():
    counts={}
    for x in CHUNKS: counts[x['source']]=counts.get(x['source'],0)+1
    top=retrieve('refund policy late delivery',3); evaluation=run_policy_evaluation()
    table,scenarios=run_scenarios(); demos=run_demos(scenarios); resume=run_resume_demo(scenarios)
    report={'chunk_counts':counts,'chunk_total':len(CHUNKS),'retrieval_top3_sources':[x['source'] for x in top],
            'policy_evaluation':evaluation,'scenario_results':table,'safe_stop_demos':demos,'crash_resume_demo':resume}
