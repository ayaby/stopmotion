#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for processing log files for the stopmotion experiment (to be
loaded by the analysis notebook)

Includes the following functions:

* process_batch - processes a single batch of participants in the 
retroactive surprise experiment. The same function
can be used for processing either a batch of the immediate 
group or a batch of the delay group.

* calc_surprise_dist - calculates the distance of each
action from the preceding/following surprising action

* process_event_seg - similar to process_batch, but processes
the event segmentation logs, not the main experiment logs

@author: Aya Ben-Yakov
"""


import itertools
import numpy as np
import pandas as pd
import json
import re
import glob
import pickle
import os.path
import scipy.signal
from scipy.stats import norm


def process_batch(group, batch_num, exp_params, file_params, overwrite=False):
    """
    Processes the logs from the stopmotion experiment

    Args:
        group (str): expected to be 'immediate' or 'delay', indicating
            which group type is being processed
        batch_num (int): batch number
        exp_params (dict): a dict with the following information about the
            experiment (same for all batches of all groups):
                n_mov (int): number of movies
                n_debrief_q (int): number of debrief questions
                first_debrief_survey (int): first debrief question that has
                    a also a text area (not just multiple choice), assumes
                    all such questions follow ones without
                batch_n_subjects (int): number of subjects per batch
        file_params (dict): a dict with information regarding input and output
            files. In all these {group} and {batch_num} will be replaced
            with the relevant variables:
                results_temp (str): template of file with the logs of the 
                    online experiment (full path)
                summary_dir (str): template of directory in which the batch 
                    summary files are saved
                debfrief_temp (str): template of the filename holding the 
                    debriefing of all subjects in the batch (fname only, will
                    be put in summary_dir)
                performance_temp (str): template of the filename holding the 
                    overall memory performance of all subjects in the batch
                    (fname only, will be put in summary_dir), divided by 
                    sections (practice, validation), no division by condition
                durations_temp (str): template of the filename holding the 
                    amount of time spent in each segment for each subject in 
                    the batch (fname only, will be put in summary_dir)
                data_dir (str): template of directory in which the batch 
                    data files are saved - these are the dataframes (tables)
                    created after processing the data. The filenames will be
                    according to each df name, in the format of
                    {group}_{batch_num}_{df_name}.tsv
                comb_data_dir (str): template of directory in which the
                    general files (same for both groups and all batches) are
                    placed
    
    Optional:
        overwrite (bool): whether to overwrite the files for this batch number
            (default False).
            
    Returns:
        all_order_idx (numpy array of int): the order index of each subj
        all_subj_ver (numpy array of int): movie version of each subj
        all_idx2num (numpy array of int): 2D array with a row for each subj
            and a column for each movie index, mapping movie index to num
        all_num2idx (numpy array of int): reverse mapping of all_idx2num,
            maps movie number to index for each subject
        surprise_ver_df (pandas DataFrame): the surprise type of each question
            for each movie version
        platform_df (pandas DataFrame): information about the platform of 
            each subject (browser, speed, etc.)
        q_info_df (pandas DataFrame): question info dataframe. Links test 
            question number to which movie/scene it was in, which q_type it 
            is, etc.
        practice_df (pandas DataFrame): results of all practice questions of 
            all subjects in a single dataframe
        validation_df (pandas DataFrame): results of all validation questions of 
            all subjects in a single dataframe
        catch_df (pandas DataFrame): results of all catch questions of 
            all subjects in a single dataframe
        test_df (pandas DataFrame): results of all test questions of 
            all subjects in a single dataframe
        assessment_df (pandas DataFrame): results of all surprise assessment
            ratings of all subjects in a single dataframe
        test_sdt_df (pandas DataFrame): similar to test_df, but the results
            of each foil are combined in one line with the results of its
            non-foil pair
        test_subj_avg_df (pandas DataFrame): performance in each condition,
            averaged over subjects
        test_q_avg_df (pandas DataFrame): performance in each condition,
            averaged over questions
    
    """

    # First checks whether this batch number has already been processed - if it
    # has and overwrite hasn't been set to True, simply loads the variables and
    # returns them instead of processing again. Assumes if one file has been
    # saved, they all have, will raise an error otherwise.
    subj_summary_fname = (file_params['summary_dir'] + \
                    file_params['performance_temp']).format(group=group, 
                    batch_num=batch_num)
    if (os.path.isfile(subj_summary_fname) and not overwrite):
        q_info_df = pd.read_csv((file_params['comb_data_dir'] + \
                                 'q_info_df.tsv'), index_col='q_num', sep='\t')
        surprise_ver_df = pd.read_csv((file_params['comb_data_dir'] + \
                                       'surprise_ver_df.tsv'), sep='\t')
        platform_df = pd.read_csv((file_params['data_dir'] + '{group}_' + 
                               '{batch_num}_{df_name}.tsv').format(group=group, 
                                batch_num=batch_num, df_name='platform_df'),
                                sep='\t')
        practice_df = pd.read_csv((file_params['data_dir'] + '{group}_' + 
                           '{batch_num}_{df_name}.tsv').format(group=group, 
                            batch_num=batch_num, df_name='practice_df'), sep='\t')
        validation_df = pd.read_csv((file_params['data_dir'] + '{group}_' + 
                           '{batch_num}_{df_name}.tsv').format(group=group, 
                            batch_num=batch_num, df_name='validation_df'), 
                            sep='\t')
        catch_df = pd.read_csv((file_params['data_dir'] + '{group}_' + 
                           '{batch_num}_{df_name}.tsv').format(group=group, 
                            batch_num=batch_num, df_name='catch_df'), sep='\t')
        test_df = pd.read_csv((file_params['data_dir'] + '{group}_' + 
                           '{batch_num}_{df_name}.tsv').format(group=group, 
                            batch_num=batch_num, df_name='test_df'), sep='\t')
        assessment_df = pd.read_csv((file_params['data_dir'] + '{group}_' + 
                           '{batch_num}_{df_name}.tsv').format(group=group, 
                            batch_num=batch_num, df_name='assessment_df'), 
                            sep='\t')
        test_sdt_df = pd.read_csv((file_params['data_dir'] + '{group}_' + 
                           '{batch_num}_{df_name}.tsv').format(group=group, 
                            batch_num=batch_num, df_name='test_sdt_df'), 
                            sep='\t')
        test_subj_avg_df = pd.read_csv((file_params['data_dir'] + '{group}_' + 
                           '{batch_num}_{df_name}.tsv').format(group=group, 
                            batch_num=batch_num, df_name='test_subj_avg_df'), 
                            sep='\t')
        test_q_avg_df = pd.read_csv((file_params['data_dir'] + '{group}_' + 
                           '{batch_num}_{df_name}.tsv').format(group=group, 
                            batch_num=batch_num, df_name='test_q_avg_df'), 
                            sep='\t')

        # Loads the additional variables from the pickle file
        subj_var_fname = file_params['data_dir'].format(group=group) + \
                            f'{group}_{batch_num}_subj_vars.pickle'
        with open(subj_var_fname, 'rb') as f:
            subj_vars = pickle.load(f)

        # Also returns the dataframes as variables
        return({'all_order_idx': subj_vars['all_order_idx'], 
                'all_subj_ver': subj_vars['all_subj_ver'], 
                'all_idx2num': subj_vars['all_idx2num'], 
                'all_num2idx': subj_vars['all_num2idx'],
                'platform_df': platform_df, 'q_info_df': q_info_df, 
                'surprise_ver_df': surprise_ver_df, 'practice_df': practice_df,
                'validation_df': validation_df, 'catch_df': catch_df,
                'test_df': test_df, 'assessment_df': assessment_df, 
                'test_sdt_df': test_sdt_df, 
                'test_subj_avg_df': test_subj_avg_df,
                'test_q_avg_df': test_q_avg_df})
        
    
    # Extracts the relevant variables from the params
    n_mov = exp_params['n_mov']
    n_debrief_q = exp_params['n_debrief_q']
    first_debrief_survey = exp_params['first_debrief_survey']
    batch_n_subjects = exp_params['batch_n_subjects']
    validation_min_Pr = exp_params['validation_min_Pr']
    catch_min_correct = exp_params['catch_min_correct']
    

    subj_durations_fname = (file_params['summary_dir'] + \
                    file_params['durations_temp']).format(group=group, 
                    batch_num=batch_num)
    all_debrief_fname = (file_params['summary_dir'] + \
                    file_params['debrief_temp']).format(group=group, 
                    batch_num=batch_num)
    results_fname = file_params['results_temp'].format(group=group, 
                    batch_num=batch_num)
    
    # Gets all components of all subjects (each component will be in separate
    # line)
    with open (results_fname, mode='r', encoding='utf-8') as res_file:
    	all_lines = list(filter(None, (line.rstrip() for line in res_file)))
      
    # Gets list of pIDs automatically from file, verifies the number of pIDS
    # matches expected
    pIDs = np.unique(np.concatenate( \
            [re.findall(r'(?:"prolific_ID":")(\w+)(?:")', line) \
             for line in all_lines]))
    
    n_subjects = len(pIDs)
    if (n_subjects != batch_n_subjects) :
      raise Exception(f'Number of unique pIDs in file ({n_subjects}) ' + \
                      f'does not match expected ({batch_n_subjects})')
    
    # Creates empty subject dataframes:
    #  Subject platform info
    #  Summary of all subjects' performance
    #  Debriefing info
    #  Practice questions
    #  Valdiation questions
    #  Catch questions
    #  Test questions

    platform_df = pd.DataFrame(columns=['subj', 'vid_ext', 'browser', 'device',
                                        'speed'])
    
    mov_idx_cols = [f'mov_idx{i}' for i in np.arange(n_mov)]
    subj_summary = pd.DataFrame(columns=['subj', 'group', 'batch', \
                                'exc_subj'] + mov_idx_cols + ['P_n_hit', 
                                'P_n_miss', 'P_Pr', 'P_n_miss_guess', 'P_n_CR',
                                'P_n_FA', 'P_n_lure_guess', 'V_n_hit', 
                                'V_n_miss', 'V_Pr', 'V_n_miss_guess', 'V_n_CR',
                                'V_n_FA', 'V_n_lure_guess', 'C_n_hit',
                                'C_n_miss', 'C_n_CR', 'C_n_FA', 'C_correct',
                                'test_n_hit', 'test_n_miss', 'test_Pr',
                                'test_n_miss_guess', 'test_n_CR', 'test_n_FA',
                                'test_n_lure_guess', 'S0', 'S1', 'S2', 'S3',
                                'N0', 'N1', 'N2', 'N3'])
    subj_summary.set_index('subj', inplace=True)
    subj_durations = pd.DataFrame(columns=['subj', 'group', 'batch', 'pID', 
                                           'practice_mov', 'practice_q',
                                           'movs', 'validation_qs', 'test_qs',
                                           'between_seg'])
    
    debrief_q_cols = [f'q{i}' for i in np.arange(1,first_debrief_survey)]
    [debrief_q_cols.extend([f'q{i}', f'q{i}_survey']) for i in \
         np.arange(first_debrief_survey,n_debrief_q+1)]    
    all_debrief = pd.DataFrame(columns=['subj', 'group', 'batch'] + \
                               debrief_q_cols + ['general_comments'])
    all_debrief.set_index('subj', inplace=True)
    
    practice_cols = ['subj', 'group', 'batch', 'q_idx', 'foil', 'correct', 
                     'guess', 'conf_level', 'rt']
    practice_df = pd.DataFrame(columns=practice_cols)
    validation_cols = ['subj', 'group', 'batch', 'mov_num', 'mov_idx', 
                       'scene_num', 'q_num', 'mov_q_idx', 'foil', 'correct',
                       'guess', 'conf_level', 'rt']
    validation_df = pd.DataFrame(columns=validation_cols)        
    catch_cols = ['subj', 'group', 'batch', 'mov_num', 'mov_idx', 'q_num', 
                  'mov_after_q_idx', 'foil', 'correct', 'guess', 'conf_level',
                  'rt']
    catch_df = pd.DataFrame(columns=catch_cols)
    test_cols = ['subj', 'group', 'batch', 'exc_subj', 'order_idx', 
                 'mov_num', 'mov_idx',
                 'scene_num', 'q_num', 'q_type', 'non_foil_q', 'target_q', 
                 'all_q_idx', 'mov_q_idx', 'surprise_type', 'foil', 'correct',
                 'guess', 'conf_level', 'rt']
    test_df = pd.DataFrame(columns=test_cols)
    assessment_cols = ['subj', 'group', 'batch', 'mov_num', 'mov_idx', 
                       'scene_num', 'q_num', 'non_foil_q', 'target_q',
                       'all_q_idx', 'mov_q_idx', 'mov_ver', 'surprise_type', 
                       'surprise_level', 'rt']
    assessment_df = pd.DataFrame(columns=assessment_cols)
    
    # List of dicts to hold more detailed segment durations for each subject 
    # (for later investigation of any outlier subjects)
    all_segment_durations = []
    
    
    # Arrays to hold on movie orders in all subjects. One for index to movie
    # num and one vice-versa. And an array for the version of each subject
    all_idx2num = np.zeros((n_subjects,n_mov))
    all_num2idx = np.zeros((n_subjects,n_mov))
    all_order_idx = np.zeros(n_subjects)
    all_subj_ver = np.zeros(n_subjects)
    
    for s in np.arange(n_subjects):
        
        # Gets current pID
        pID = pIDs[s]

        # Gets the subset of subject lines
        subj_lines = [l for l in all_lines if pID in l]

        # First extracts the exp_info component (that also includes info on
        # number of movies, used for verification)
        exp_info_idx = np.nonzero(['"result_type":"start_exp"' in str \
                                 for str in subj_lines])[0]
        if (len(exp_info_idx) != 1):
            raise Exception(f'{s}-{pID} should have exactly one exp info comp')
        exp_info_idx = exp_info_idx[0]
        exp_info_comp = json.loads(subj_lines[exp_info_idx])[0]
        
        # Gets speed and browser/device info and adds to platform_info_df.
        # Delay group will have two browser info tests - takes only study
        # Browser info may not exist in some of the earlier subjects, if it 
        # doesn't - defines as Unknown. One subject doesn't have 'result_type'
        # defined so doesn't use that to identify the speed/browser comps
        browser_idx = np.nonzero(['"device_type":' in str \
                                 for str in subj_lines])[0]
        browser_idx = browser_idx[0]
        browser_comp = json.loads(subj_lines[browser_idx])[0]
        if ('browser' in browser_comp.keys()):
            browser = browser_comp['browser']
        else:
            browser = 'Unknown'
            
        speed_idx = np.nonzero(['"speed":' in str \
                                 for str in subj_lines])[0]
        speed_idx = speed_idx[0]
        speed_comp = json.loads(subj_lines[speed_idx])[0]

        platform_df = platform_df.append({'subj': s, 
                            'vid_ext': exp_info_comp['vid_ext'][1:],
                            'browser': browser, 
                            'device': browser_comp['device_type'], 
                            'speed': speed_comp['speed']}, ignore_index=True)
        
        # Saves the movie order index and the subj version of each subject
        all_order_idx[s] = exp_info_comp['subj_mov_order_idx']
        all_subj_ver[s] = exp_info_comp['subjVer']
    
        n_questions = exp_info_comp['n_questions']
        n_validation_qs = exp_info_comp['n_validation_qs']
        #n_catch_qs = exp_info_comp['n_catch_qs']                  
        
        # Gets the indices of the different component types, verifying there's
        # exactly the expected number of matches

        mov_comp_idx  = np.nonzero(['"result_type":"film_viewing"' in str \
                                    for str in subj_lines])[0]
        practice_idx = np.nonzero(['"result_type":"practice_scene"' in str \
                                 for str in subj_lines])[0] 
        q_comp_idx = np.nonzero(['"result_type":"test_questions"' in str \
                                 for str in subj_lines])[0]    
        assessment_idx = np.nonzero(['"result_type":"surprise_assessment"' \
                                 in str for str in subj_lines])[0]
        debrief_idx = np.nonzero(['"result_type":"debrief"' in str \
                                 for str in subj_lines])[0]

        if (len(mov_comp_idx) != n_mov):
            raise Exception(f'{s}-{pID} has incorrect number of film comps')
        if (len(q_comp_idx) != 1):
            raise Exception(f'{s}-{pID} should have exactly one questions comp')
        if (len(practice_idx) != 1):
            raise Exception(f'{s}-{pID} should have exactly one practice comp')
        if (len(assessment_idx) != 1):
            raise Exception(f'{s}-{pID} should have exactly one surprise ' +
                            'assessment comp')
        # At least one subject doesn't have a debrief component, so allows 
        # having no debrief component
        if (len(debrief_idx) > 1):
            raise Exception(f'{s}-{pID} should have at most one debrief comp')
        
        q_comp_idx = q_comp_idx[0]
        practice_idx = practice_idx[0]
        assessment_idx = assessment_idx[0]
        
        # Converts the json string in each component into a dict, divided by type

        mov_comps = []
        mov_validation_idx = []
        for m in mov_comp_idx:
            mov_comp = json.loads(subj_lines[m])
            mov_comps.append(mov_comp)
            mov_validation_idx.append(np.flatnonzero([('trial_name' in c) and \
                                       (c['trial_name']=='validation_question') \
                                       for c in mov_comp]))
        practice_comp = json.loads(subj_lines[practice_idx])
        questions_comp = json.loads(subj_lines[q_comp_idx])
        assessment_comp = json.loads(subj_lines[assessment_idx])
        
        if (len(debrief_idx)>0):
            debrief_idx = debrief_idx[0]
            debrief_comp = json.loads(subj_lines[debrief_idx])[0]
        else:
            debrief_comp = None
          
          
        # Extracts movie order and creates conversions from movie index to num
        # Also adds to arrays with all subjects
        mov_idx2num = np.array(exp_info_comp['movie_order'])
        mov_num2idx = np.zeros((n_mov), dtype=int)
        mov_num2idx[mov_idx2num] = np.arange(n_mov)
        
        all_idx2num[s,] = mov_idx2num
        all_num2idx[s,] = mov_num2idx
    
        # Gets number of practce questions from practice comp
        n_practice_qs = practice_comp[0]['n_practice_qs']
        
        
        #
        # Calculates the time each part took, to later identify participants that
        # took long breaks. Durations are initially in msec, converts to sec
        #
        
        
        curr_durations = {}
        curr_durations['vid_press_play'] = np.array(
                [mov_comp[0]['vid_start_time'] - mov_comp[0]['vid_loaded_time'] \
                 for mov_comp in mov_comps])/1000
    
        curr_durations['vid_press_cont'] = np.array(
                [mov_comp[0]['vid_pressed_cont_time'] - \
                 mov_comp[0]['vid_ended_time'] for mov_comp in mov_comps])/1000
        
        # Calculates time for the practice session
        curr_durations['practice_press_play'] = \
                                (practice_comp[0]['vid_start_time'] - \
                                 practice_comp[0]['vid_loaded_time'])/1000
    
        curr_durations['practice_press_cont'] = \
                                (practice_comp[0]['vid_pressed_cont_time'] - \
                                 practice_comp[0]['vid_ended_time'])/1000
        curr_durations['practice_q_total'] = \
                                (practice_comp[0]['practice_q_end_time'] - \
                                 practice_comp[0]['vid_pressed_cont_time'])/1000
            
        # Saves both the per-question rt for all validation questions of each 
        # movie, the total time spent on validation questions, and the time spent
        # on each post-validation screen
        # The mov_num fields are ordered by mov_num instead of the presentation 
        # order to facilitate comparison across subjects
        curr_durations['mov_validation_qs'] = \
            [[mov_comp[i]['rt']/1000 for i in mov_validation_idx[m]] \
                 for m, mov_comp in enumerate(mov_comps)]
        curr_durations['mov_validation_total'] = \
            [sum(curr_durations['mov_validation_qs'][m]) \
             for m in np.arange(n_mov)]
        curr_durations['mov_num_validation_qs'] = \
            [curr_durations['mov_validation_qs'][m] for m in mov_num2idx]
        curr_durations['mov_num_validation_total'] = \
            [curr_durations['mov_validation_total'][m] for m in mov_num2idx]
        curr_durations['mov_post_validation'] = \
            list(itertools.chain.from_iterable([[trial['rt']/1000 \
            for trial in mov_comp if ('transition_screen' in trial and \
            trial['transition_screen']=='post_validation_q')] \
            for mov_comp in mov_comps]))
        
        # Saves total time on questions, and total time per movie, in addition
        # to time on pre and post test screens
        pre_trials = [q for q in questions_comp if ('transition_screen' in q and \
                        q['transition_screen']=='pre_test_q')]
        post_trials = [q for q in questions_comp if ('transition_screen' in q and \
                        q['transition_screen']=='post_test_q')]
        curr_durations['mov_pre_test'] = [trial['rt']/1000 \
                                              for trial in pre_trials]
        curr_durations['mov_post_test'] = [trial['rt']/1000 \
                                              for trial in post_trials]
        curr_durations['mov_test_total'] = \
            [sum([q['rt']/1000 for q in questions_comp if ('mov_idx' in q and \
                        q['mov_idx']==m+1)]) for m in np.arange(n_mov)]
        curr_durations['mov_num_test_total'] = \
            [sum([q['rt']/1000 for q in questions_comp if ('mov_num' in q and \
                        q['mov_num']==m+1)]) for m in np.arange(n_mov)]
        
        # Saves total time on surprise assessment questions and time on final
        # transition screen
        curr_durations['assessment_q_total'] = \
            sum([q['rt']/1000 for q in assessment_comp if ('mov_idx' in q )])
        curr_durations['post_assessment'] = \
            sum([q['rt']/1000 for q in assessment_comp if ('transition_screen' \
                 in q and q['transition_screen']=='post_assessment')])
        
        # Adds to list of durations of all subjects and calculates more general
        # timings to add to the duration dataframe
        all_segment_durations.append(curr_durations)
        total_practice_mov = practice_comp[0]['vid_pressed_cont_time']/1000 - \
                    practice_comp[0]['vid_loaded_time']/1000
        total_mov = sum([mov_comp[0]['vid_pressed_cont_time'] - \
                              mov_comp[0]['vid_loaded_time'] \
                              for mov_comp in mov_comps])/1000
        total_validation = sum(curr_durations['mov_validation_total'])
        total_test = sum(curr_durations['mov_test_total'])
        total_between = sum(curr_durations['mov_post_validation']) + \
            sum(curr_durations['mov_pre_test']) + \
            sum(curr_durations['mov_post_test']) + \
            curr_durations['post_assessment']
        subj_durations = subj_durations.append({'subj':s, 
                            'group': group[0].upper(), 'batch': batch_num,
                            'pID':pID, 'practice_mov':total_practice_mov, 
                            'practice_q':curr_durations['practice_q_total'],
                            'movs':total_mov, 'validation_qs':total_validation,
                            'test_qs':total_test, 'assessment_qs': \
                            curr_durations['assessment_q_total'], 
                            'between_seg':total_between}, ignore_index=True)
    
        #
        # Adds info from debrief component to the dataframs with debriefing of
        # all subjects. At least one subject didn't finish debrief so adds a 
        # line only if the debrief_comp exists
        #
        if (debrief_comp is not None):
            debrief_dict = {'subj':s, 'group': group[0].upper(), 
                            'batch': batch_num,
                            'general_comments':debrief_comp['general_comments']}
            debrief_dict.update({f'q{i}': debrief_comp[f'q{i}']['answer'] \
                            for i in np.arange(1,n_debrief_q+1)})
            debrief_dict.update({f'q{i}_survey': debrief_comp[f'q{i}']['survey'] \
                            for i in np.arange(first_debrief_survey,
                            n_debrief_q+1)})
            all_debrief = all_debrief.append(debrief_dict, ignore_index=True)
    
    
        #
        # Aggregates all questions (practice, validation, test) into separate
        # dataframes with the results of all subjects. In calculating Pr - 
        # assumes number of foils equals number of non-foils
        #
        
        # Adds all practice questions to all-subj dataframe and collects
        # info about the number of hits/miss etc. for the summary df

        practice_q_idx = np.flatnonzero([('trial_name' in c) and \
                                       (c['trial_name']=='practice_question') \
                                       for c in practice_comp])
        p_df = pd.DataFrame([practice_comp[i] for i in practice_q_idx])
        p_df['subj'] = s
        p_df['group'] = group[0].upper()
        p_df['batch'] = batch_num
        practice_df = practice_df.append(p_df.loc[:,practice_cols], 
                                                ignore_index=True)
        practice_n_hit = sum((p_df['foil']==False) & \
                               (p_df['correct']==True))
        practice_n_miss = sum((p_df['foil']==False) & \
                               (p_df['correct']==False))
        practice_n_miss_guess = sum((p_df['foil']==False) & \
                               (p_df['guess']==True))
        practice_n_CR = sum((p_df['foil']==True) & \
                               (p_df['correct']==True))
        practice_n_FA = sum((p_df['foil']==True) & \
                               (p_df['correct']==False) & \
                               (p_df['guess']==False))
        practice_n_lure_guess = sum((p_df['foil']==True) & \
                               (p_df['guess']==True))   
        practice_Pr = (practice_n_hit-practice_n_FA)/(n_practice_qs/2)
        
        
        # Adds all validation questions to all-subj dataframe and collects
        # info about the number of hits/miss etc. for the summary df
        for m in np.arange(n_mov): 
            mov_comp = mov_comps[mov_num2idx[m]]
            v_df = pd.DataFrame([mov_comp[i] for i in \
                                   mov_validation_idx[mov_num2idx[m]]])
            v_df['subj'] = s
            v_df['group'] = group[0].upper()
            v_df['batch'] = batch_num
            validation_df = validation_df.append(\
                                v_df.loc[:,validation_cols].sort_values(by=\
                                ['subj', 'mov_num', 'scene_num', 'q_num'],
                                axis=0), ignore_index=True)
        validation_n_hit = sum((validation_df['subj']==s) & \
                               (validation_df['foil']==False) & \
                               (validation_df['correct']==True))
        validation_n_miss = sum((validation_df['subj']==s) & \
                                (validation_df['foil']==False) & \
                               (validation_df['correct']==False))
        validation_n_miss_guess = sum((validation_df['subj']==s) & \
                                      (validation_df['foil']==False) & \
                                      (validation_df['guess']==True))
        validation_n_CR = sum((validation_df['subj']==s) & \
                              (validation_df['foil']==True) & \
                               (validation_df['correct']==True))
        validation_n_FA = sum((validation_df['subj']==s) & \
                              (validation_df['foil']==True) & \
                               (validation_df['correct']==False) & \
                               (validation_df['guess']==False))
        validation_n_lure_guess = sum((validation_df['subj']==s) & \
                                      (validation_df['foil']==True) & \
                                      (validation_df['guess']==True))            
        validation_Pr = (validation_n_hit-validation_n_FA)/(n_validation_qs/2)

          
        # Adds all catch questions to all-subj dataframe and collects
        # info about the number of hits/miss etc. for the summary df
        catch_q_idx = np.flatnonzero([('trial_name' in c) and \
                                       (c['trial_name']=='catch_question') \
                                       for c in questions_comp])
        c_df = pd.DataFrame([questions_comp[i] for i in catch_q_idx])
        c_df['subj'] = s
        c_df['group'] = group[0].upper()
        c_df['batch'] = batch_num
        catch_df = catch_df.append(c_df.loc[:,catch_cols], 
                                                ignore_index=True)
        catch_n_hit = sum((c_df['foil']==False) & \
                               (c_df['correct']==True))
        catch_n_miss = sum((c_df['foil']==False) & \
                               (c_df['correct']==False))
        catch_n_CR = sum((c_df['foil']==True) & \
                               (c_df['correct']==True))
        catch_n_FA = sum((c_df['foil']==True) & \
                               (c_df['correct']==False) & \
                               (c_df['guess']==False))
        catch_correct = catch_n_hit + catch_n_CR


        # Checks whether the subject should be excluded based on validation
        # and/or catch questions, and marks included subjects in the test
        # question dataframe (currently doesn't remove them, just identifies
        # them so they can be removed prior to analysis)
        exc_subj = False
        if ((validation_Pr<validation_min_Pr) or \
            (catch_correct<catch_min_correct)):
            exc_subj = True
        
        # Adds all test questions to all-subj dataframe and collects
        # info about the number of hits/miss etc. for the summary df. Conf
        # levels were accidentally reversed when saving logs, so highest conf 
        # is -2 instead of 2
        q_index = np.flatnonzero([('trial_name' in c) and \
                                   (c['trial_name']=='test_question') \
                                   for c in questions_comp])
        q_df = pd.DataFrame([questions_comp[i] for i in q_index])
        q_df['subj'] = s
        q_df['group'] = group[0].upper()
        q_df['batch'] = batch_num
        q_df['exc_subj'] = exc_subj
        q_df['order_idx'] = all_order_idx[s]
        test_df = test_df.append(\
                            q_df.loc[:,test_cols].sort_values(by=\
                            ['mov_num', 'scene_num', 'q_num'],
                            axis=0), ignore_index=True)
        test_n_hit = sum((q_df['foil']==False) & (q_df['correct']==True))
        test_n_HC_hit = sum((q_df['foil']==False) & (q_df['correct']==True) & \
                            (q_df['conf_level']==-2))
        test_n_miss = sum((q_df['foil']==False) & (q_df['correct']==False))
        test_n_miss_guess = sum((q_df['foil']==False) & (q_df['guess']==True))
        test_n_CR = sum((q_df['foil']==True) & (q_df['correct']==True))
        test_n_FA = sum((q_df['foil']==True) & (q_df['correct']==False) & \
                        (q_df['guess']==False))
        test_n_HC_FA = sum((q_df['foil']==True) & (q_df['correct']==False) & \
                            (q_df['guess']==False) & (q_df['conf_level']==-2))
        test_n_lure_guess = sum((q_df['foil']==True) & (q_df['guess']==True))  
        test_Pr = (test_n_hit-test_n_FA)/(n_questions/2)
        test_HC_Pr = (test_n_HC_hit-test_n_HC_FA)/(n_questions/2)
        

        # Adds all assessment questions to all-subj dataframe and collects
        # info about the number of each surprise rating for the summary df
        # Also adds movie version to the dataframe
        assessment_q_idx = np.flatnonzero([('trial_name' in c) and \
                                       (c['trial_name']=='surprise_assessment') \
                                       for c in assessment_comp])
        a_df = pd.DataFrame([assessment_comp[i] for i in assessment_q_idx])
        a_df['subj'] = s
        a_df['group'] = group[0].upper()
        a_df['batch'] = batch_num
        a_df['mov_ver'] = all_subj_ver[s]
        assessment_df = assessment_df.append(a_df.loc[:,assessment_cols], 
                                                ignore_index=True)
        
        assessment_S0 = sum((a_df['surprise_type']=='S') & \
                               (a_df['surprise_level']==0))
        assessment_S1 = sum((a_df['surprise_type']=='S') & \
                               (a_df['surprise_level']==1))
        assessment_S2 = sum((a_df['surprise_type']=='S') & \
                               (a_df['surprise_level']==2))
        assessment_S3 = sum((a_df['surprise_type']=='S') & \
                               (a_df['surprise_level']==3))
        assessment_N0 = sum((a_df['surprise_type']=='N') & \
                               (a_df['surprise_level']==0))
        assessment_N1 = sum((a_df['surprise_type']=='N') & \
                               (a_df['surprise_level']==1))
        assessment_N2 = sum((a_df['surprise_type']=='N') & \
                               (a_df['surprise_level']==2))
        assessment_N3 = sum((a_df['surprise_type']=='N') & \
                               (a_df['surprise_level']==3))
        
        # Adds all performance summary variables to the summary dataframe. Also
        # adds movie order. 
        summary_dict = {'subj':s, 'group': group[0].upper(), 'batch': batch_num,
                        'exc_subj': exc_subj, 'P_n_hit':practice_n_hit,  
                        'P_n_miss':practice_n_miss, 'P_Pr':practice_Pr,  
                        'P_n_miss_guess':practice_n_miss_guess, 
                        'P_n_CR':practice_n_CR, 'P_n_FA':practice_n_FA, 
                        'P_n_lure_guess':practice_n_lure_guess,
                        'V_n_hit':validation_n_hit, 
                        'V_n_miss':validation_n_miss, 'V_Pr':validation_Pr, 
                        'V_n_miss_guess':validation_n_miss_guess, 
                        'V_n_CR':validation_n_CR, 'V_n_FA':validation_n_FA, 
                        'V_n_lure_guess':validation_n_lure_guess,
                        'C_n_hit':catch_n_hit, 
                        'C_n_miss':catch_n_miss, 'C_n_CR':catch_n_CR, 
                        'C_n_FA':catch_n_FA, 'C_correct':catch_correct,
                        'test_n_hit':test_n_hit, 'test_n_miss':test_n_miss,
                        'test_n_miss_guess':test_n_miss_guess, 
                        'test_n_CR':test_n_CR, 'test_n_FA':test_n_FA, 
                        'test_n_lure_guess':test_n_lure_guess,
                        'test_Pr':test_Pr, 'test_HC_Pr':test_HC_Pr,
                        'S0':assessment_S0, 'S1':assessment_S1, 
                        'S2':assessment_S2, 'S3':assessment_S3, 
                        'N0':assessment_N0, 'N1':assessment_N1, 
                        'N2':assessment_N2, 'N3':assessment_N3}
        for m in np.arange(n_mov):
            summary_dict.update({f'mov_idx{m}':mov_idx2num[m]})
        subj_summary = subj_summary.append(pd.DataFrame.from_records(\
                                    [summary_dict], index='subj'), sort=False)
    
    # Updates the relevant fields in the different dataframes that were 
    # created to be int
    subj_summary = subj_summary.astype({'P_n_hit': 'int', 'P_n_miss': 'int', 
                                        'P_Pr': 'float64', 'P_n_miss_guess': 'int',
                                        'P_n_CR': 'int', 'P_n_FA': 'int', 
                                        'P_n_lure_guess': 'int', 
                                        'V_n_hit': 'int', 'V_n_miss': 'int',
                                        'V_Pr': 'float64', 
                                        'V_n_miss_guess': 'int', 
                                        'V_n_CR': 'int', 'V_n_FA': 'int', 
                                        'V_n_lure_guess': 'int', 
                                        'C_n_hit': 'int', 'C_n_miss': 'int', 
                                        'C_n_CR': 'int', 'C_n_FA': 'int', 
                                        'C_correct': 'int', 
                                        'test_n_hit': 'int', 'test_n_miss': 'int', 
                                        'test_Pr': 'float64',
                                        'test_n_miss_guess': 'int', 
                                        'test_n_CR': 'int', 'test_n_FA': 'int',
                                        'test_n_lure_guess': 'int', 'S0': 'int', 
                                        'S1': 'int', 'S2': 'int', 
                                        'S3': 'int', 'N0': 'int', 
                                        'N1': 'int', 'N2': 'int',
                                        'N3': 'int'})
        
    practice_df = practice_df.astype({'conf_level': 'int', 'rt': 'int'})
    validation_df = validation_df.astype({'conf_level': 'int', 'rt': 'int'})
    catch_df = catch_df.astype({'conf_level': 'int', 'rt': 'int'})
    test_df = test_df.astype({'conf_level': 'int', 'rt': 'int'})
    assessment_df = assessment_df.astype({'surprise_level': 'int', 'rt': 'int'})

    # For all question types reverses confidence level which was
    # accidentally saved reversed (-2 for high confidence it occurred)
    practice_df['conf_level'] = -practice_df['conf_level']
    validation_df['conf_level'] = -validation_df['conf_level']
    catch_df['conf_level'] = -catch_df['conf_level']
    test_df['conf_level'] = -test_df['conf_level']
    
    # Changes surprise type in test_df to be based on the target (currently
    # all non-targets are listed as neutral), and adds mov_ver and surprise
    # level from the assessment_df
    test_df = test_df.drop(columns=['surprise_type']).merge(assessment_df[\
                              ['subj', 'surprise_type', 'target_q', 'mov_ver',
                               'surprise_level']], validate='many_to_one')
     

    # Adds the number of preceding surprising scenes for each target to the 
    # surprise assessment df and to the test_df. The first part of the sum
    # counts the number of surprising events in earlier movies
    # and the second part of the sum counts the number of surprising events
    # preceding current one within the same movie (this is calculated per
    # row and added as a column)
    assessment_df['num_prev_S'] = assessment_df.apply(lambda row: sum(
                                (assessment_df['subj']==row['subj'])&\
                                (assessment_df['mov_idx']<row['mov_idx'])&\
                                (assessment_df['surprise_type']=='S')) + \
                                sum((assessment_df['subj']==row['subj'])&\
                                (assessment_df['mov_idx']==row['mov_idx'])&\
                                (assessment_df['target_q']<row['target_q'])&\
                                (assessment_df['surprise_type']=='S')), axis=1)
    test_df = test_df.merge(assessment_df[['subj', 'target_q', 'num_prev_S']], 
                            validate='many_to_one')

    # Creates a df of q_num to mov info (same for all subjects). This info
    # is currently also in the separate dfs, but this allows it to be removed
    # from some of the other dfs. 
    q_info_df = test_df[['mov_num', 'scene_num', 'q_type', 'foil', 'q_num', 
                         'non_foil_q', 'target_q']].drop_duplicates().\
                         set_index('q_num')
    
    
    # Adds a column to test_df indicating the action index (when the action
    # was presented for each order). First calculates the per-movie question
    # number (non-foils only). then creates a df of the movie number and
    # index (presentation order) for each order. Uses these to calculate
    # the index of each action - the presentation order of that action 
    # (with foils and non-foils receiving the same number as they refer to
    # the same action)
    n_q_per_mov = q_info_df.groupby(['mov_num'])['non_foil_q'].nunique()
    q_info_df['mov_non_foil_q'] = [(r.non_foil_q+1)/2-sum(n_q_per_mov\
             .loc[:r.mov_num-1]) for r in q_info_df.itertuples()]
    order_df = test_df[['order_idx', 'mov_idx', 'mov_num']].drop_duplicates()
            
    test_df['action_idx'] = test_df.apply(lambda row: sum(n_q_per_mov.loc[\
           order_df[(order_df['order_idx']==row.order_idx) & \
                    (order_df['mov_idx']<row.mov_idx)]['mov_num']]) + \
        q_info_df.loc[row.q_num]['mov_non_foil_q'], axis=1)
        
    
    # Creates a dataframe with the surprise type of each target for each movie
    # version.
    surprise_ver_df = assessment_df[['mov_ver', 'mov_num', 'scene_num', 'q_num',
                                   'target_q', 'surprise_type']].\
                                   drop_duplicates().set_index('q_num').\
                                   sort_values(['mov_ver', 'mov_num', 
                                    'target_q'])

    # Creates a combined dataframe that holds the hit and FA for each pair
    # of questions, as well as the surprise level from the surprise assessment 
    # phase. For FA uses confidence level as guesses aren't considered correct.
    # Adds columns for miss and nFA (not FA - which includes guesses, or low
    # confidence for the HC analysis)
    # Adds also columns for high confidence (HC) hits and FA used to calculate
    # high-confidence Pr
    test_hits_df = test_df[test_df['foil']==False].drop(columns=['q_num', 
                          'all_q_idx', 'mov_q_idx', 'mov_num', 'mov_idx', \
                          'scene_num', 'foil', 'guess', 'rt']).rename(\
                            columns={'correct':'hit', 'conf_level': 'conf'})
    test_hits_df['hit'] = test_hits_df['hit'].astype('int')
    test_hits_df['HC_hit'] = (test_hits_df['conf']==2).astype('int')
    test_hits_df['miss'] = np.logical_not(test_hits_df['hit']).astype('int')
    test_hits_df['HC_miss'] = np.logical_not(test_hits_df['HC_hit']).\
                                astype('int')
    test_FA_df = test_df[test_df['foil']==True].drop(columns=['q_num', \
                        'all_q_idx', 'mov_q_idx', 'mov_num', 'mov_idx', \
                        'scene_num', 'foil', 'guess', 'rt']).rename(columns=\
                        {'conf_level': 'lure_conf'})
    test_FA_df['FA'] = test_FA_df['lure_conf'].apply(lambda x: int(x>0))
    test_FA_df['HC_FA'] = (test_FA_df['lure_conf']==2).astype('int')
    test_FA_df['nFA'] = np.logical_not(test_FA_df['FA']).astype('int')
    test_FA_df['HC_nFA'] = np.logical_not(test_FA_df['HC_FA']).\
                                astype('int')
    test_sdt_df = test_hits_df.merge(test_FA_df, validate='one_to_one')
    test_sdt_df['Pr'] = test_sdt_df['hit']-test_sdt_df['FA']
    test_sdt_df['HC_Pr'] = test_sdt_df['HC_hit']-test_sdt_df['HC_FA']
    
    # Gets the number of hits/FA/Pr per subject for each question type X 
    # surprise type, as well as hit rate and FA rate for calculating d'
    test_subj_avg_df = test_sdt_df.groupby(['subj', 'group', 'batch', 
                            'exc_subj', 'order_idx', 'mov_ver', 'q_type', 
                            'surprise_type']).agg({'hit': 'sum', 'HC_hit': \
                            'sum', 'miss': 'sum', 'HC_miss': 'sum',
                            'conf': 'mean', 'FA': 'sum', 'HC_FA': 'sum', 
                            'nFA': 'sum', 'HC_nFA': 'sum', 
                            'lure_conf': 'mean', 'Pr': 'mean', 
                            'HC_Pr': 'mean'}).reset_index().rename(columns=\
                            {'hit':'n_hit', 'HC_hit':'n_HC_hit', 
                             'miss':'n_miss', 'HC_miss':'n_HC_miss', 
                             'conf':'avg_conf', 'FA':'n_FA', 'HC_FA':'n_HC_FA',
                             'nFA':'n_nFA', 'HC_nFA':'n_HC_nFA',
                             'lure_conf': 'avg_lure_conf'})
   
    
    # Adds a d' column, with adjustment to avoid infinity (uses the Macmillan &
    # Kaplan 1985 approach, adjusting only extreme rates 
    
    # First gets the number of hits, misses, etc.
    nH = np.array(test_subj_avg_df['n_hit']).astype(float)
    nM = np.array(test_subj_avg_df['n_miss']).astype(float)
    nHC_H = np.array(test_subj_avg_df['n_HC_hit']).astype(float)
    nHC_M = np.array(test_subj_avg_df['n_HC_miss']).astype(float)
    nFA = np.array(test_subj_avg_df['n_FA']).astype(float)
    n_nFA = np.array(test_subj_avg_df['n_nFA']).astype(float)
    nHC_FA = np.array(test_subj_avg_df['n_HC_FA']).astype(float)
    nHC_nFA = np.array(test_subj_avg_df['n_HC_nFA']).astype(float)
    
    # Then calculates hit and FA rate, adjusting all the values that will 
    # lead to a rate of 0/1
    H_rate = nH/(nH+nM)
    HC_H_rate = nHC_H/(nHC_H+nHC_M)
    FA_rate = nFA/(nFA+n_nFA)
    HC_FA_rate = nHC_FA/(nHC_FA+nHC_nFA)
   
    H_rate[H_rate==0] = 0.5/(nH[H_rate==0]+nM[H_rate==0])
    H_rate[H_rate==1] = (nH[H_rate==1]+nM[H_rate==1]-0.5)/(nH[H_rate==1]+\
                          nM[H_rate==1])
    HC_H_rate[HC_H_rate==0] = 0.5/(nHC_H[HC_H_rate==0]+nHC_M[HC_H_rate==0])
    HC_H_rate[HC_H_rate==1] = (nHC_H[HC_H_rate==1]+nHC_M[HC_H_rate==1]-0.5)/\
                        (nHC_H[HC_H_rate==1]+nHC_M[HC_H_rate==1])
    FA_rate[FA_rate==0] = 0.5/(nFA[FA_rate==0]+n_nFA[FA_rate==0])
    FA_rate[FA_rate==1] = (nFA[FA_rate==1]+n_nFA[FA_rate==1]-0.5)/\
                        (nFA[FA_rate==1]+n_nFA[FA_rate==1])
    HC_FA_rate[HC_FA_rate==0] = 0.5/(nHC_FA[HC_FA_rate==0]+\
                          nHC_nFA[HC_FA_rate==0])
    HC_FA_rate[HC_FA_rate==1] = (nHC_FA[HC_FA_rate==1]+\
                      nHC_nFA[HC_FA_rate==1]-0.5)/(nHC_FA[HC_FA_rate==1]+\
                             nHC_nFA[HC_FA_rate==1])
                        
    
    # Calculates d' from the adjusted hit and FA rates
    test_subj_avg_df['d'] = norm.ppf(H_rate) - norm.ppf(FA_rate)
    test_subj_avg_df['HC_d'] = norm.ppf(HC_H_rate) - \
                                    norm.ppf(HC_FA_rate)


    # Similar df, but aggregating over questions instead of subjects. Removes
    # excluded subjects prior to averaging
    test_q_avg_df = test_sdt_df[(test_sdt_df['exc_subj']==False)].groupby(\
                                  ['non_foil_q', 'surprise_type']).agg(\
                                    {'hit': 'sum', 'HC_hit':'sum', 
                                     'conf': 'mean', 'FA': 'sum', 
                                     'HC_FA':'sum', 'lure_conf': 'mean', 
                                     'Pr': 'mean', 'HC_Pr': 'mean'}).\
                                    reset_index().rename(columns={\
                                    'hit':'n_hit', 'HC_hit':'n_HC_hit', 
                                    'FA': 'n_FA', 'HC_FA': 'n_HC_FA', 
                                    'conf': 'avg_conf', 
                                    'lure_conf': 'avg_lure_conf'})
         
    
    # Saves the summary dataframes to files
    all_debrief.to_csv(all_debrief_fname, sep='\t')
    subj_summary.to_csv(subj_summary_fname, sep='\t')
    subj_durations.set_index('subj', inplace=True)
    subj_durations.to_csv(subj_durations_fname, sep='\t')

                           
    # Saves dataframes to files that can then be loaded by the notebook. 
    # q_info and surprise_ver are saved only for the first batch, since they'll
    # be identical for all batches (also same for both groups - second group
    # will overwrite first
    if (batch_num==1):
        q_info_df.to_csv((file_params['comb_data_dir'] + '{df_name}.tsv').\
                         format(group=group, df_name='q_info_df'), sep='\t')
        surprise_ver_df.to_csv((file_params['comb_data_dir'] + 
                            '{df_name}.tsv').format(group=group, 
                            df_name='surprise_ver_df'), sep='\t', index=False)
    platform_df.to_csv((file_params['data_dir'] + '{group}_{batch_num}_' + 
                           '{df_name}.tsv').format(group=group, 
                            batch_num=batch_num, df_name='platform_df'),
                            sep='\t', index=False)
    practice_df.to_csv((file_params['data_dir'] + '{group}_{batch_num}_' + 
                       '{df_name}.tsv').format(group=group, 
                        batch_num=batch_num, df_name='practice_df'), sep='\t',
                        index=False)
    validation_df.to_csv((file_params['data_dir'] + '{group}_{batch_num}_' + 
                       '{df_name}.tsv').format(group=group, 
                        batch_num=batch_num, df_name='validation_df'), 
                        sep='\t', index=False)
    catch_df.to_csv((file_params['data_dir'] + '{group}_{batch_num}_' + 
                       '{df_name}.tsv').format(group=group, 
                        batch_num=batch_num, df_name='catch_df'), sep='\t',
                        index=False)
    test_df.to_csv((file_params['data_dir'] + '{group}_{batch_num}_' + 
                       '{df_name}.tsv').format(group=group, 
                        batch_num=batch_num, df_name='test_df'), sep='\t',
                        index=False)
    assessment_df.to_csv((file_params['data_dir'] + '{group}_{batch_num}_' + 
                       '{df_name}.tsv').format(group=group, 
                        batch_num=batch_num, df_name='assessment_df'), 
                        sep='\t', index=False)
    test_sdt_df.to_csv((file_params['data_dir'] + '{group}_{batch_num}_' + 
                       '{df_name}.tsv').format(group=group, 
                        batch_num=batch_num, df_name='test_sdt_df'), 
                        sep='\t', index=False)
    test_subj_avg_df.to_csv((file_params['data_dir'] + '{group}_{batch_num}_' + 
                       '{df_name}.tsv').format(group=group, 
                        batch_num=batch_num, df_name='test_subj_avg_df'), 
                        sep='\t', index=False)
    test_q_avg_df.to_csv((file_params['data_dir'] + '{group}_{batch_num}_' + 
                       '{df_name}.tsv').format(group=group, 
                        batch_num=batch_num, df_name='test_q_avg_df'), 
                        sep='\t', index=False)

    # Saves all additional variables in a single pickle file
    subj_vars = {'all_order_idx': all_order_idx, 'all_subj_ver': all_subj_ver, 
            'all_idx2num': all_idx2num, 'all_num2idx': all_num2idx}
    subj_var_fname = file_params['data_dir'].format(group=group) + \
                            f'{group}_{batch_num}_subj_vars.pickle'
    with open(subj_var_fname, 'wb') as f:
        pickle.dump(subj_vars, f)

    # Also returns the dataframes as variables
    return({'all_order_idx': all_order_idx, 'all_subj_ver': all_subj_ver, 
            'all_idx2num': all_idx2num, 'all_num2idx': all_num2idx,
            'platform_df': platform_df, 'q_info_df': q_info_df, 
            'surprise_ver_df': surprise_ver_df, 'practice_df': practice_df,
            'validation_df': validation_df, 'catch_df': catch_df,
            'test_df': test_df, 'assessment_df': assessment_df, 
            'test_sdt_df': test_sdt_df, 
            'test_subj_avg_df': test_subj_avg_df,
            'test_q_avg_df': test_q_avg_df})
    

def calc_surprise_dist(action_time_fname, surprise_ver_df, q_info_df, 
                       data_dir, n_ver=4, n_mov=3, overwrite=False):
    """
    Calculates the distance of each action from the previous/next surprise
    
    Creates a dataframe that holds for each version of each movie, the distance
    between each action and the closest preceding/following surprising one (if
    a question is before the first surprise or after the last one, this will be
    -1). Used to calculate the secondary analysis with a linear mixed model.
    This function must be run after process_batch has been run at least once,
    since it relys of files created by it.
    
    Args:
        action_time_fname (str): name of file with the time of each action (in
            csv format, can be read as dataframe)
        surprise_ver_df (DataFrame): a df (created by process_batch) with an
            indication of whether each target question is in its 
            surprising/neutral version in each movie version
        q_info_df (DataFrame): a df (created by process_batch) with 
            basic per-question information
        data_dir (str): directory in which to save the surprise_dist dataframe
            (the file will be called surprise_dist_df.csv)
        
    Optional:
        overwrite (bool): whether to overwrite the surprise_dist file. If False
            (default), will load the dataframe from the existing file instead 
            of calculating it.
        n_ver (int): number of versions (default 4, the number of versions used
            in the experiment). Should not be changed.
        n_mov (int): number of movies (default 3, the number of movies in the
            experiment). Should not be changed.
        
    Returns:
        surprise_dist_df (DataFrame): a df that holds for each question in each
            movie (under each version) - the question number, time and distance
            of the closest preceding/following surprising action
    """
    
    # First checks whether the dataframe has already been calculated and saved 
    # to a file - if it has and overwrite hasn't been set to True, loads and
    # returns the dataframe from the file
    surprise_dist_fname = data_dir + '/surprise_dist_df.tsv'
    if (os.path.isfile(surprise_dist_fname) and not overwrite):
        surprise_dist_df = pd.read_csv(surprise_dist_fname, sep='\t')
        return surprise_dist_df
    
    
    action_time_df = pd.read_csv(action_time_fname)

    all_ver = []
    for mov_ver in np.arange(n_ver)+1:
        for mov_num in np.arange(n_mov):
    
            # Gets a list of all target questions that appeared in their
            # surprising version, for the current movie in the current version
            ver_surprise_T = np.array(surprise_ver_df[\
                                    (surprise_ver_df['mov_ver']==mov_ver)&\
                                    (surprise_ver_df['surprise_type']=='S')&\
                                    (surprise_ver_df['mov_num']==mov_num)]\
                                    ['target_q'])
            min_q = min(q_info_df[q_info_df['mov_num']==mov_num].index)
            max_q = max(q_info_df[q_info_df['mov_num']==mov_num].index)
            
            # Divides all questions (including neutral and non-target) into
            # groups according to the closest preceding/following surprising
            # action. First calculates the number of questions in each 
            # group and then uses that to create arrays with the question num
            # of the preceding/following surprising target of each question.
            # For the next surprise - calculates as though the foil questions
            # are the surprising ones (when calculating the number of questions
            # per group) to ensure that foils and non-foils get the same 
            # question number as the next surprising action
            n_q_per_group_P = [ver_surprise_T[0]-min_q]+list(\
                              ver_surprise_T[1:]-ver_surprise_T[0:-1])+\
                              [max_q-ver_surprise_T[-1]+1]
            n_q_per_group_F = [ver_surprise_T[0]-min_q+2]+list(\
                              ver_surprise_T[1:]-ver_surprise_T[0:-1])+\
                              [max_q-ver_surprise_T[-1]-1]
            preceding_S = np.concatenate([np.tile(v,s) for (s,v) in zip (\
                            n_q_per_group_P, [-1]+list(ver_surprise_T))], None)
            following_S = np.concatenate([np.tile(v,s) for (s,v) in zip (\
                            n_q_per_group_F, list(ver_surprise_T)+[-1])], None)
            
            #
            # Creates a dataframe with the information for the current movie
            # in the current version. For each question it holds the time of 
            # the corresponding action in the movie and the number and time of
            # the previous/next surprising action. Will have -1/Nan for 
            # questions that come before the first surprise/after the last one 
            # in the corresponding fields (-1 for number, NaN for time). Also
            # adds for each question the distance from the previous/next 
            # surprise (this field will be used as a predictor in the analysis)
            #
            
            # Creates a dataframe with the prev/next surprise question num, 
            # here only includes the target questions
            curr_ver_df = pd.DataFrame.from_dict({\
                            'q_num':np.arange(min_q,max_q+1), 'mov_ver':mov_ver,
                            'mov_num':mov_num, 'prev_S_num':preceding_S, 
                            'next_S_num':following_S})
            
            # Adds information about the time of each question, using both the
            # q_info_df for all questions matching each target and the 
            # action_time_df to get the time of each question
            curr_ver_df = curr_ver_df.merge(q_info_df[['foil', 'non_foil_q', 
                            'target_q']].astype({'target_q':'int'}).reset_index(),
                            on='q_num', validate='one_to_one').astype(\
                            {'non_foil_q':'int'}).merge(action_time_df[\
                            ['non_foil_q', 'time_msec']], on=['non_foil_q'], 
                            validate='many_to_one').rename(columns={\
                            'time_msec':'action_time'})
            
            # Adds the time of the prev and next target, using action_time_df
            curr_ver_df = curr_ver_df.merge(action_time_df[['non_foil_q', \
                                'time_msec']].rename(columns={\
                                'non_foil_q':'prev_S_num'}), 'left').rename(\
                                columns={'time_msec':'prev_S_time'})
            curr_ver_df = curr_ver_df.merge(action_time_df[['non_foil_q', \
                                'time_msec']].rename(columns={\
                                'non_foil_q':'next_S_num'}), 'left').rename(\
                                columns={'time_msec':'next_S_time'})
            
            # Calculates the distance between each action and the prev/next
            # surprise and adds as new columns
            curr_ver_df['prev_S_dist'] = curr_ver_df['action_time'] - \
                                            curr_ver_df['prev_S_time']
            curr_ver_df['next_S_dist'] = curr_ver_df['next_S_time'] - \
                                            curr_ver_df['action_time']
            
            # Adds to a list of all dataframes (all movies in all versions)
            all_ver.append(curr_ver_df)
        
    # Concatenates into a single dataframe, saves to a file and returns
    surprise_dist_df = pd.concat(all_ver).reset_index(drop=True)
    surprise_dist_df.to_csv(surprise_dist_fname, sep='\t', index=False)
    return surprise_dist_df


def process_event_seg(seg_params, seg_file_params):
    """
    Processes the event segmentation logs from the stopmotion experiment

    Args:
        seg_params (dict): a dict with the following information:
                demo_boundaries (np.array(int)): timing of the boundaries in 
                    the demo movie in msec
                demo_length (int): duration of demo in msec
                mov_scene_change (np.array(int)): timing of the scene changes
                    in each film
                mov_lengths (np.array(int)): duration of each film (msec)
                n_debrief_q (int): number of debrief questions
                first_debrief_survey (int): first debrief question that has
                    a also a text area (not just multiple choice), assumes
                    all such questions follow ones without
                boundary_min_dist (int): minimal distance between button 
                    presses to be considered two boundaries
                win_size (int): sliding window size (sec) for boundary 
                    aggregation analysis
                boundary_min_subj (int): minimal number of participants
                    for a peak to be considered a boundary in the
                    sliding window analysis
                jump (int): jump (sec) for sliding window analysis
                
        seg_file_params (dict): a dict with information regarding input and 
            output files. In the results template, uses '*' to find 
            data_dir (str): template of directory in which the data files
                are saved - these are the dataframes (tables)
                created after processing the data.   
            results_temp (str): template of file with the event seg logs
            summary_fname (str): file where summary performance is saved
            debfrief_fname (str): name of file holding the debriefing
                of all subjects
            durations_fname (str): name of file holding the amount of 
                time spent in each segment for each subject 
            
    Returns:

        all_idx2num (numpy array of int): 2D array with a row for each subj
            and a column for each movie index, mapping movie index to num
        all_num2idx (numpy array of int): reverse mapping of all_idx2num,
            maps movie number to index for each subject
        practice_df (pandas DataFrame): results of all practice questions of 
            all subjects in a single dataframe
        validation_df (pandas DataFrame): results of all validation questions 
            of all subjects in a single dataframe
        test_df (pandas DataFrame): results of all test questions of 
            all subjects in a single dataframe
        summary_df (pandas DataFrame) : dataframe with a summary of memory
            performance per subject
        subj_boundaries (pandas DataFrame): timings of all boundaries of all
            subjects, linked to the nearest scene change
        subj_mean_boundaries (pandas DataFrame): timings of mean boundaries
            over subjects (peaks of sliding window analysis), linked to the
            nearest scene change
        all_peak_heights (numpy array of int): the heights of the peaks
            from the sliding window analysis (the number of participants
            who identified a boundary in each bin)
    
    """

    # Gets list of filenames from the results template 
    all_fnames = glob.glob(seg_file_params['data_dir'] + \
                           seg_file_params['results_temp'])
    

    n_mov = len(seg_params['mov_lengths'])
    n_subjects = len(all_fnames)
    
    
    # Creates empty subject dataframes:
    #  Summary of all subjects' performance
    #  Summary of timings
    #  All boundaries of all films
    #  Debriefing info
    #  Practice questions
    #  Valdiation questions
    #  Test questions
    mov_idx_cols = [f'mov_idx{i}' for i in np.arange(n_mov)]
    subj_summary = pd.DataFrame(columns=['sub'] + mov_idx_cols + \
                                ['practice_n_hit', 'practice_n_miss', 
                                 'practice_Pr', 'practice_n_miss_guess', 
                                 'practice_n_CR', 'practice_n_FA', 
                                 'practice_n_lure_guess', 'validation_n_hit', 
                                 'validation_n_miss', 'validation_Pr', 
                                 'validation_n_miss_guess', 'validation_n_CR', 
                                 'validation_n_FA', 'validation_n_lure_guess',
                                 'test_n_hit', 'test_n_miss', 'test_Pr',
                                 'test_n_miss_guess', 'test_n_CR', 
                                 'test_n_FA', 'test_n_lure_guess'])
    subj_summary.set_index('sub', inplace=True)
    subj_durations = pd.DataFrame(columns=['sub', 'pID', 'practice_mov', 
                                           'practice_q', 'movs', 
                                           'validation_qs', 'test_qs',
                                           'between_seg'])
    subj_boundaries = pd.DataFrame(columns=['sub', 'mov_num', 'time',
                                            'orig_time', 'mov_scene', 
                                            'prev_mov_scene', 'scene_dist',
                                            'prev_scene_dist'])

    debrief_q_cols = [f'q{i}' for i in np.arange(1, \
                      seg_params['first_debrief_survey'])]
    [debrief_q_cols.extend([f'q{i}', f'q{i}_survey']) for i in \
         np.arange(seg_params['first_debrief_survey'], \
                   seg_params['n_debrief_q']+1)]    
    all_debrief = pd.DataFrame(columns=['sub'] + debrief_q_cols + \
                               ['general_comments'])
    
    practice_cols = ['sub', 'q_idx', 'foil', 'correct', 'guess', 'conf_level',
                     'rt']
    practice_df = pd.DataFrame(columns=practice_cols)
    validation_cols = ['sub', 'mov_num', 'mov_idx', 'scene_num', 'q_num', 
                       'mov_q_idx', 'foil', 'correct', 'guess', 'conf_level',
                       'rt']
    validation_df = pd.DataFrame(columns=validation_cols)
    test_cols = ['sub', 'mov_num', 'mov_idx', 'scene_num', 'q_num', 
                       'q_type', 'non_foil_q', 'target_q', 'all_q_idx',
                       'mov_q_idx', 'foil', 'correct', 'guess', 'conf_level',
                       'rt']
    test_df = pd.DataFrame(columns=test_cols)
    
    
    # List of dicts to hold more detailed segment durations for each subject 
    # (for later investigation of any outlier subjects)
    all_segment_durations = []
    
    # Arrays to hold on movie orders in all subjects. One for index to movie
    # num and one vice-versa
    all_idx2num = np.zeros((n_subjects,n_mov))
    all_num2idx = np.zeros((n_subjects,n_mov))
    
    for s in np.arange(n_subjects):
        
        # Loads all the components as separate lines
        results_fname = all_fnames[s]
        with open (results_fname, mode='r', encoding='utf-8') as res_file:
            	lines = list(filter(None, (line.rstrip() for line in res_file)))
        
        # Extracts the pID from the file and as a sanity check, verifies there 
        # is only one 
        line_IDs = np.unique(np.concatenate( \
                    [re.findall(r'(?:"prolific_ID":")(\w+)(?:")', line) \
                     for line in lines]))
        if (len(line_IDs)>1):
            raise Exception(results_fname + ' has more than one pID')
        pID = line_IDs[0]
        
        
        # Gets the indices of the different component types, verifying there's
        # exactly the expected number of matches
        
        mov_comp_idx  = np.nonzero(['"result_type":"mov_boundaries"' in str \
                                    for str in lines])[0]      
        demo_mov_idx = mov_comp_idx[0] # Assummes demo is always first movie
        q_comp_idx = np.nonzero(['"result_type":"test_questions"' in str \
                                 for str in lines])[0]    
        practice_idx = np.nonzero(['"result_type":"practice_scene"' in str \
                                 for str in lines])[0]        
        debrief_idx = np.nonzero(['"result_type":"debrief"' in str \
                                 for str in lines])[0]
        
        if (len(mov_comp_idx) != n_mov+1):
            raise Exception(f'{s}-{pID} has incorrect number of mov comps')
        if (len(q_comp_idx) != 1):
            raise Exception(f'{s}-{pID} should have exactly one questions comp')
        if (len(practice_idx) != 1):
            raise Exception(f'{s}-{pID} should have exactly one practice comp')
        if (len(debrief_idx) != 1):
            raise Exception(f'{s}-{pID} should have exactly one debrief comp')
        
        q_comp_idx = q_comp_idx[0]
        practice_idx = practice_idx[0]
        debrief_idx = debrief_idx[0]
        
        
        # Converts the json string in each component into a dict, divided by type
        demo_comp = json.loads(lines[demo_mov_idx])
        
        mov_comps = []
        mov_validation_idx = []
        for m in mov_comp_idx[1:]:
            mov_comp = json.loads(lines[m])
            mov_comps.append(mov_comp)
            mov_validation_idx.append(np.flatnonzero([('trial_name' in c) and \
                                       (c['trial_name']=='validation_question') \
                                       for c in mov_comp]))
        practice_comp = json.loads(lines[practice_idx])
        questions_comp = json.loads(lines[q_comp_idx])
        debrief_comp = json.loads(lines[debrief_idx])
        
        
        # Extracts movie order and creates conversions from movie index to num
        # Also adds to arrays with all subjects
        mov_idx2num = np.array([mov_comp[0]['mov_num'] for mov_comp in mov_comps])
        mov_num2idx = np.zeros((n_mov), dtype=int)
        mov_num2idx[mov_idx2num-1] = np.arange(n_mov)
        
        all_idx2num[s,] = mov_idx2num
        all_num2idx[s,] = mov_num2idx
    
        #
        # Calculates the time each part took, to later identify participants that
        # took long breaks. Durations are initially in msec, converts to sec
        #
        
        
        curr_durations = {}
        curr_durations['vid_press_play'] = np.array(
                [mov_comp[0]['vid_start_time'] - mov_comp[0]['vid_loaded_time'] \
                 for mov_comp in mov_comps])/1000
    
        curr_durations['vid_press_cont'] = np.array(
                [mov_comp[0]['vid_pressed_cont_time'] - \
                 mov_comp[0]['vid_ended_time'] for mov_comp in mov_comps])/1000
        
        # Calculates time for the practice session
        curr_durations['practice_press_play'] = \
                                (practice_comp[0]['vid_start_time'] - \
                                 practice_comp[0]['vid_loaded_time'])/1000
    
        curr_durations['practice_press_cont'] = \
                                (practice_comp[0]['vid_pressed_cont_time'] - \
                                 practice_comp[0]['vid_ended_time'])/1000
        curr_durations['practice_q_total'] = \
                                (practice_comp[0]['practice_q_end_time'] - \
                                 practice_comp[0]['vid_pressed_cont_time'])/1000
            
        # Saves both the per-question rt for all validation questions of each 
        # movie, the total time spent on validation questions, and the time spent
        # on each post-validation screen
        # The mov_num fields are ordered by mov_num instead of the presentation 
        # order to facilitate comparison across subjects
        curr_durations['mov_validation_qs'] = \
            [[mov_comp[i]['rt']/1000 for i in mov_validation_idx[m]] \
                 for m, mov_comp in enumerate(mov_comps)]
        curr_durations['mov_validation_total'] = \
            [sum(curr_durations['mov_validation_qs'][m]) \
             for m in np.arange(n_mov)]
        curr_durations['mov_num_validation_qs'] = \
            [curr_durations['mov_validation_qs'][m] for m in mov_num2idx]
        curr_durations['mov_num_validation_total'] = \
            [curr_durations['mov_validation_total'][m] for m in mov_num2idx]
        curr_durations['mov_post_validation'] = \
            list(itertools.chain.from_iterable([[trial['rt']/1000 \
            for trial in mov_comp if ('transition_screen' in trial and \
            trial['transition_screen']=='post_validation_q')] \
            for mov_comp in mov_comps]))
        
        # Saves total time on questions, and total time per movie, in addition
        # to time on pre and post test screens
        pre_trials = [q for q in questions_comp if ('transition_screen' in q and \
                        q['transition_screen']=='pre_test_q')]
        post_trials = [q for q in questions_comp if ('transition_screen' in q and \
                        q['transition_screen']=='post_test_q')]
        curr_durations['mov_pre_test'] = [trial['rt']/1000 \
                                              for trial in pre_trials]
        curr_durations['mov_post_test'] = [trial['rt']/1000 \
                                              for trial in post_trials]
        curr_durations['mov_test_total'] = \
            [sum([q['rt']/1000 for q in questions_comp if ('mov_idx' in q and \
                        q['mov_idx']==m+1)]) for m in np.arange(n_mov)]
        curr_durations['mov_num_test_total'] = \
            [sum([q['rt']/1000 for q in questions_comp if ('mov_num' in q and \
                        q['mov_num']==m+1)]) for m in np.arange(n_mov)]
        
        # Adds to list of durations of all subjects and calculates more general
        # timings to add to the duration dataframe
        all_segment_durations.append(curr_durations)
        total_practice_mov = practice_comp[0]['vid_pressed_cont_time']/1000 - \
                    practice_comp[0]['vid_loaded_time']/1000
        total_mov = sum([mov_comp[0]['vid_pressed_cont_time'] - \
                              mov_comp[0]['vid_loaded_time'] \
                              for mov_comp in mov_comps])/1000
        total_validation = sum(curr_durations['mov_validation_total'])
        total_test = sum(curr_durations['mov_test_total'])
        total_between = sum(curr_durations['mov_post_validation']) + \
            sum(curr_durations['mov_pre_test']) + \
            sum(curr_durations['mov_post_test'])
        subj_durations = subj_durations.append({'sub':s, 'pID':pID,
                                'practice_mov':total_practice_mov, 
                                'practice_q':curr_durations['practice_q_total'],
                                'movs':total_mov, 'validation_qs':total_validation,
                                'test_qs':total_test, 'between_seg':total_between},
                                ignore_index=True)
    
    
        #
        # Extracts subject boundaries for each movie and adds to a general
        # dataframe. Also calculates the closest matching scene change (once 
        # absolute closest and once the closest preceding one). Saves
        # both the original recorded boundary time, and the adjusted time
        # (when subtracting estimated RT)
        #
        
        # First calculates the time between each actual color change in the
        # demo and the logged press.
        curr_demo_boundaries = np.array(demo_comp['boundaries'])
        sub_demo_dist = []
        for demo_b,next_b in zip(seg_params['demo_boundaries'], \
                            np.append(seg_params['demo_boundaries'][1:], \
                              seg_params['demo_length'])):
            
            # For each colour change - takes the minimal press following it,
            # as long as it precedes the next colour change
            sect_presses = curr_demo_boundaries[(curr_demo_boundaries>=demo_b)&\
                (curr_demo_boundaries<=next_b)]
            if (sect_presses.size>0):
                sub_demo_dist.append(min(sect_presses-demo_b))
        subj_rt = np.mean(sub_demo_dist) 
        
        
        for m in np.arange(n_mov):
            
            curr_boundaries = np.array(mov_comps[mov_num2idx[m]][0]\
                                       ['boundaries'])
            
            # Due to current programming, pressing space before the video
            # starts will be logged as a boundary with a very long time.
            # Removes all boundaries at the beginning that are larger than 
            # following ones (starts with minimal value boundary) and removes
            # all boundaries after film end.
            max_boundary = mov_comps[mov_num2idx[m]][0]['vid_ended_time'] - \
                                mov_comps[mov_num2idx[m]][0]['vid_start_time']
            curr_boundaries = curr_boundaries[np.argmin(curr_boundaries):]
            curr_boundaries = curr_boundaries[np.where(curr_boundaries <= \
                                                       max_boundary)[0]]
            
            # Any boundaries identified within 2s of the previous boundary
            # are treated as the same boundary (the border screen changes for
            # 1s, plus reaction time)
            curr_boundaries = np.delete(curr_boundaries, 
                                np.where((curr_boundaries[1:] - \
                                curr_boundaries[:-1]) < \
                                seg_params['boundary_min_dist'])[0]+1)
            
            curr_scene_change = np.array([0] + \
                                         seg_params['mov_scene_change'][m])
            # Identifies the closest scene change
            curr_boundaries_match = curr_scene_change[np.argmin(\
                                        [np.array(abs(curr_boundaries-b)) \
                                         for b in curr_scene_change], axis=0)]
            # Identifiesn the closest scene change that preceded the boundary
            # (here taking the original button press, not corrected, to 
            # allow for slight differences in RT)
            curr_boundaries_match_prev = curr_scene_change[np.argmin(\
                        [np.array(curr_boundaries<b-subj_rt)*max_boundary + \
                         np.multiply(curr_boundaries>=b-subj_rt, \
                        abs(curr_boundaries-b)) for b in curr_scene_change], \
                        axis=0)]
            curr_boundaries_df = pd.DataFrame({'sub':s, 'mov_num':m+1, 
                                'time':curr_boundaries-subj_rt,
                                'orig_time':curr_boundaries, 
                                'mov_scene':curr_boundaries_match,
                                'prev_mov_scene':curr_boundaries_match_prev})
            curr_boundaries_df['scene_dist'] = curr_boundaries_df['time']-\
                                            curr_boundaries_df['mov_scene']
            curr_boundaries_df['prev_scene_dist'] = \
                                    curr_boundaries_df['time']- \
                                    curr_boundaries_df['prev_mov_scene']
            subj_boundaries = subj_boundaries.append(curr_boundaries_df, 
                                                     ignore_index=True)
    
        #
        # Adds info from debrief component to the dataframs with debriefing of
        # all subjects
        #
        debrief_dict = {'sub':s, 
                        'general_comments':debrief_comp['general_comments']}
        debrief_dict.update({f'q{i}': debrief_comp[f'q{i}']['answer'] \
                        for i in np.arange(1,seg_params['n_debrief_q']+1)})
        debrief_dict.update({f'q{i}_survey': debrief_comp[f'q{i}']['survey'] \
                        for i in np.arange(seg_params['first_debrief_survey'],
                                           seg_params['n_debrief_q']+1)})
        all_debrief = all_debrief.append(debrief_dict, ignore_index=True)
    
    
        #
        # Aggregates all questions (practice, validation, test) into separate
        # dataframes with the results of all subjects
        #
        
        # Adds all practice questions to all-subj dataframe and collects
        # info about the number of hits/miss etc. for the summary df

        practice_q_idx = np.flatnonzero([('trial_name' in c) and \
                                       (c['trial_name']=='practice_question') \
                                       for c in practice_comp])
        p_df = pd.DataFrame([practice_comp[i] for i in practice_q_idx])
        p_df['sub'] = s
        practice_df = practice_df.append(p_df.loc[:,practice_cols], 
                                                ignore_index=True)
        practice_n_hit = sum((p_df['foil']==False) & \
                               (p_df['correct']==True))
        practice_n_miss = sum((p_df['foil']==False) & \
                               (p_df['correct']==False))
        practice_n_miss_guess = sum((p_df['foil']==False) & \
                               (p_df['guess']==True))
        practice_n_CR = sum((p_df['foil']==True) & \
                               (p_df['correct']==True))
        practice_n_FA = sum((p_df['foil']==True) & \
                               (p_df['correct']==False) & \
                               (p_df['guess']==False))
        practice_n_lure_guess = sum((p_df['foil']==True) & \
                               (p_df['guess']==True))   
            
            
        # Adds all validation questions to all-subj dataframe and collects
        # info about the number of hits/miss etc. for the summary df
        for m in np.arange(n_mov): 
            mov_comp = mov_comps[mov_num2idx[m]]
            v_df = pd.DataFrame([mov_comp[i] for i in \
                                   mov_validation_idx[mov_num2idx[m]]])
            v_df['sub'] = s
            validation_df = validation_df.append(\
                                v_df.loc[:,validation_cols].sort_values(by=\
                                ['sub', 'mov_num', 'scene_num', 'q_num'],
                                axis=0), ignore_index=True)
            
        validation_n_hit = sum((validation_df['sub']==s) & \
                               (validation_df['foil']==False) & \
                               (validation_df['correct']==True))
        validation_n_miss = sum((validation_df['sub']==s) & \
                                (validation_df['foil']==False) & \
                               (validation_df['correct']==False))
        validation_n_miss_guess = sum((validation_df['sub']==s) & \
                                      (validation_df['foil']==False) & \
                                      (validation_df['guess']==True))
        validation_n_CR = sum((validation_df['sub']==s) & \
                              (validation_df['foil']==True) & \
                               (validation_df['correct']==True))
        validation_n_FA = sum((validation_df['sub']==s) & \
                              (validation_df['foil']==True) & \
                               (validation_df['correct']==False) & \
                               (validation_df['guess']==False))
        validation_n_lure_guess = sum((validation_df['sub']==s) & \
                                      (validation_df['foil']==True) & \
                                      (validation_df['guess']==True))            
            
        # Adds all test questions to all-subj dataframe and collects
        # info about the number of hits/miss etc. for the summary df

        q_index = np.flatnonzero([('trial_name' in c) and \
                                   (c['trial_name']=='test_question') \
                                   for c in questions_comp])
        q_df = pd.DataFrame([questions_comp[i] for i in q_index])
        q_df['sub'] = s
        test_df = test_df.append(\
                            q_df.loc[:,test_cols].sort_values(by=\
                            ['mov_num', 'scene_num', 'q_num'],
                            axis=0), ignore_index=True)
        test_n_hit = sum((q_df['foil']==False) & \
                               (q_df['correct']==True))
        test_n_miss = sum((q_df['foil']==False) & \
                               (q_df['correct']==False))
        test_n_miss_guess = sum((q_df['foil']==False) & \
                               (q_df['guess']==True))
        test_n_CR = sum((q_df['foil']==True) & \
                               (q_df['correct']==True))
        test_n_FA = sum((q_df['foil']==True) & \
                               (q_df['correct']==False) & \
                               (q_df['guess']==False))
        test_n_lure_guess = sum((q_df['foil']==True) & \
                               (q_df['guess']==True))  
        
    
        # Adds all performance summary variables to the summary dataframe. Also
        # adds movie order
        summary_dict = {'sub':s, 'practice_n_hit':practice_n_hit, 
                        'practice_n_miss':practice_n_miss, 
                        'practice_n_miss_guess':practice_n_miss_guess, 
                        'practice_n_CR':practice_n_CR, 
                        'practice_n_FA':practice_n_FA, 
                        'practice_n_lure_guess':practice_n_lure_guess,
                        'practice_Pr':practice_n_hit-practice_n_FA, 
                        'validation_n_hit':validation_n_hit, 
                        'validation_n_miss':validation_n_miss,
                        'validation_n_miss_guess':validation_n_miss_guess, 
                        'validation_n_CR':validation_n_CR, 
                        'validation_n_FA':validation_n_FA, 
                        'validation_n_lure_guess':validation_n_lure_guess,
                        'validation_Pr':validation_n_hit-validation_n_FA, 
                        'test_n_hit':test_n_hit, 'test_n_miss':test_n_miss,
                        'test_n_miss_guess':test_n_miss_guess, 
                        'test_n_CR':test_n_CR, 'test_n_FA':test_n_FA, 
                        'test_n_lure_guess':test_n_lure_guess,
                        'test_Pr':test_n_hit-test_n_FA, }
        for m in np.arange(n_mov):
            summary_dict.update({f'mov_idx{m}':mov_idx2num[m]})
        subj_summary = subj_summary.append(pd.DataFrame.from_records(\
                                    [summary_dict], index='sub'), sort=False)

    # Runs a sliding window (3-sec window with 1-sec jumps) and calculates the
    # number of subjects who identified a boundary (if one subject pressed
    # multiple times that won't count). For each of the identified boundaries,
    # matches with the closest scene change and calculates the distance, 
    # aggregating in a dataframe.
    mov_comb_boundaries = []
    all_peak_heights = []
    subj_mean_boundaries = pd.DataFrame(columns=['mov_num','time',
                                                 'mov_scene','scene_dist'])
    for m in np.arange(n_mov):
        curr_boundaries = subj_boundaries[subj_boundaries['mov_num']==m+1]
        bin_n_boundaries = \
            np.array([len(np.unique(curr_boundaries[(curr_boundaries['time']>=\
            max(0,(s-seg_params['win_size']/2)*1000)) & \
            (curr_boundaries['time']<=min(seg_params['mov_lengths'][m],
                (s+seg_params['win_size']/2)*1000))]['sub'])) \
            for s in np.arange(0,seg_params['mov_lengths'][m]/1000+1)])
        peaks = scipy.signal.find_peaks(bin_n_boundaries, 1)
        all_peak_heights.extend(peaks[1]['peak_heights'])
        curr_boundaries_mean = peaks[0][peaks[1]['peak_heights'] > \
                                seg_params['boundary_min_subj']]*1000
        mov_comb_boundaries.append(curr_boundaries_mean)

        # Aggregates in a dataframe
        curr_scene_change = np.array(seg_params['mov_scene_change'][m])
        curr_boundaries_match = curr_scene_change[np.argmin( \
                                    [np.array(abs(curr_boundaries_mean-b)) \
                                     for b in curr_scene_change], axis=0)]

        tmp_df = pd.DataFrame({'mov_num':m+1, 'time':curr_boundaries_mean,
                                'mov_scene':curr_boundaries_match, 
                                'scene_dist':curr_boundaries_mean-\
                                curr_boundaries_match})
        # Adds the original scene changes to the dataframe, in case any weren't
        # identified by participants. These will appear with NaN in the time
        # column
        tmp_df = tmp_df.merge(pd.DataFrame({'mov_scene': curr_scene_change}),
                              how='outer', sort=True)  

        subj_mean_boundaries = subj_mean_boundaries.append(tmp_df, 
                                    ignore_index=True)
        
    
    # Saves the summarizing dfs to files
    all_debrief.to_csv(seg_file_params['debrief_fname'], sep='\t')
    subj_summary.to_csv(seg_file_params['summary_fname'], sep='\t')
    subj_durations.to_csv(seg_file_params['durations_fname'], sep='\t')
    
    # Returns the performance dfs and the boundary df
    return({'all_idx2num': all_idx2num, 'all_num2idx': all_num2idx,
            'practice_df': practice_df, 'validation_df': validation_df,
            'test_df': test_df, 'summary_df': subj_summary,
            'subj_boundaries': subj_boundaries, 
            'subj_mean_boundaries': subj_mean_boundaries,
            'all_peak_heights': all_peak_heights})

