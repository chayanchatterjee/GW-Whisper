#! /usr/bin/env python

#Basic imports
import argparse
import numpy as np
from queue import Queue
import h5py
import os
import logging

#PyCBC imports
from pycbc.types import TimeSeries

#BnsLib imports
from bnslib import *
from bnslib import progress_tracker

SEC_PER_MONTH = 30 * 24 * 60 * 60

def get_start_time(fn):
    start = int(fn.split('-')[1])
    if start == 0:
        return start
    else:
        return start + 0.1


def split_true_and_false_positives(event_list, injection_times,
                                   tolerance=3., assume_sorted=False,
                                   workers=0):
    """Find a list of correctly identified events.
    
    Arguments
    ---------
    event_list : list of tuple of float
        A list of events as returned by get_event_list.
    injection_times : numpy.array of floats
        An array containing the times at which a signal was actually
        present in the data.
    tolerance : {float, 3.}
        The maximum time in seconds an injection time may be away from
        an event time to be counted as a true positive.
    assume_sorted : {bool, False}
        Assume that the injection_times are sorted in an ascending
        order. (If this is false the injection times are sorted
        internally)
    workers : {int or None, 0}
        How many processes to use to split the events. If set to 0, the
        events are analyzed sequentially. If set to None spawns as many
        processes as there are CPUs available.
    
    Returns
    -------
    true_positives : list of tuples of float
        A list of events that were correctly identified as events.
    false_positives : list of tuples of float
        A list of events that were falsely identified as events.
    """
    if assume_sorted:
        injtimes = injection_times
    else:
        injtimes = injection_times.copy()
        injtimes.sort()

    def worker(sub_event_list, itimes, tol, output, wid):
        tp = []
        fp = []
        for event in sub_event_list:
            t, v = event
            idx = np.searchsorted(itimes, t, side='right')
            if idx == 0:
                diff = abs(t - itimes[0])
            elif idx == len(itimes):
                diff = abs(t - itimes[-1])
            else:
                diff = min(abs(t - itimes[idx-1]), abs(t - itimes[idx]))
            if diff <= tol:
                tp.append(event)
            else:
                fp.append(event)
        output.put((wid, tp, fp))

    if workers == 0:
        queue = Queue()
        worker(event_list, injtimes, tolerance, queue, 0)
        _, tp, fp = queue.get()
        return tp, fp
    else:
        if workers is None:
            workers = mp.cpu_count()
        idxsrange = int(len(event_list) // workers)
        overhang = len(event_list) - workers * idxsrange
        prev = 0
        queue = mp.Queue()
        jobs = []
        for i in range(workers):
            if i < overhang:
                end = prev + idxsrange + 1
            else:
                end = prev + idxsrange
            p = mp.Process(target=worker,
                           args=(event_list[prev:end],
                                 injtimes,
                                 tolerance,
                                 queue,
                                 i))
            prev = end
            jobs.append(p)

        for p in jobs:
            p.start()

        results = [queue.get() for p in jobs]

        for p in jobs:
            p.join()

        results = sorted(results, key=lambda inp: inp[0])
        tp = []
        fp = []
        for res in results:
            tp.extend(res[1])
            fp.extend(res[2])
        return tp, fp

def get_event_times(event_list):
    """Extract the event times from a list of events.
    
    Arguments
    ---------
    event_list : list of tuples of float
        A list of events as returned by get_event_list.
    
    Returns
    -------
    list of float
        A list containing the times of the events given by the
        event_list.
    """
    return [event[0] for event in event_list]

def get_closest_injection_times(injection_times, times,
                                return_indices=False,
                                assume_sorted=False):
    """Return a list of the closest injection times to a list of input
    times.
    
    Arguments
    ---------
    injection_times : numpy.array of floats
        An array containing the times at which a signal was actually
        present in the data.
    times : iterable of floats
        A list of times. The function checks which injection time was
        closest to every single one of these times.
    return_indices : {bool, False}
        Return the indices of the found injection times.
    assume_sorted : {bool, False}
        Assume that the injection times are sorted in ascending order.
        (If set to false, the injection times are sorted internally)
    
    Returns
    -------
    numpy.array of float:
        Returns an array containing the injection times that were
        closest to the provided times. The order is given by the order
        of the input times.
    numpy.array of int, optional:
        Return an array of the corresponding indices. (Only returned if
        return_indices is true)
    """
    if assume_sorted:
        injtimes = injection_times
        sidxs = np.arange(len(injtimes))
    else:
        sidxs = injection_times.argsort()
        injtimes = injection_times[sidxs]

    ret = []
    idxs = []
    for t in times:
        idx = np.searchsorted(injtimes, t, side='right')
        if idx == 0:
            ret.append(injtimes[idx])
            idxs.append(sidxs[idx])
        elif idx == len(injtimes):
            ret.append(injtimes[idx-1])
            idxs.append(sidxs[idx-1])
        else:
            if abs(t - injtimes[idx-1]) < abs(t - injtimes[idx]):
                idx -= 1
            ret.append(injtimes[idx])
            idxs.append(sidxs[idx])
    if return_indices:
        return np.array(ret), np.array(idxs, dtype=int)
    else:
        return np.array(ret)

def get_missed_injection_times(event_list, injection_times,
                               tolerance=3., return_indices=False):
    """Find the injection times that are not present in a provided list
    of events.
    
    Arguments
    ---------
    event_list : list of tuples of float
        A list of events as returned by get_event_list.
    injection_times : numpy.array of floats
        An array containing the times at which a signal was actually
        present in the data.
    tolerance : {float, 3.}
        The maximum time in seconds an injection time may be away from
        an event time to be counted as a true positive.
    return_indices : {bool, False}
        Return the indices of the missed injection times.
    
    Returns
    -------
    numpy.array of floats:
        Returns an array containing injection times that were not
        contained in the list of events, considering the tolerance.
    numpy.array of int, optional:
        Return an array of the corresponding indices. (Only returned if
        return_indices is true)
    """
    ret = []
    idxs = []
    event_times = np.array(get_event_times(event_list))
    if len(event_times) == 0:
        return injection_times
    for idx, inj_time in enumerate(injection_times):
        if np.min(np.abs(event_times - inj_time)) > tolerance:
            ret.append(inj_time)
            idxs.append(idx)
    if return_indices:
        return np.array(ret), np.array(idxs, dtype=int)
    else:
        return np.array(ret)
    



def get_event_times(event_list):
    """Extract the event times from a list of events.
    
    Arguments
    ---------
    event_list : list of tuples of float
        A list of events as returned by get_event_list.
    
    Returns
    -------
    list of float
        A list containing the times of the events given by the
        event_list.
    """
    return [event[0] for event in event_list]



def get_cluster_boundaries(triggers, boundarie_time=1.):
    """A basic clustering algorithm that generates a list start and end
    times for every cluster.
    
    Arguments
    ---------
    triggers : iterable of floats or 2D array
        A list or array containing the times of a time series that
        exceed a given threshold. (As returned by get_trigger_times or
        get_triggers)
    boundarie_time : {float, 1.}
        A time in seconds around the cluster boundaries that may not
        contain any triggers for the cluster to be complete.
    
    Returns
    -------
    list of list of float:
        Returns a list that contains the boundarie times of all
        clusters. As such each entry is a list of length 2. The first
        of which is the inital time of the cluster, the second is the
        final time of the cluster.
    
    Note
    ----
    This is a very basic clustering algorithm that simply expands the
    boundaries of all clusters until there are no triggers within an
    accepted range.
    """
    if np.ndim(triggers) == 1:
        trigger_times = triggers
    elif np.ndim(triggers) == 2:
        trigger_times = triggers[0]
    else:
        raise RuntimeError
    i = 0
    clusters = []
    current_cluster = []
    while i < len(trigger_times):
        if len(current_cluster) == 0:
            current_cluster.append(trigger_times[i])
        elif len(current_cluster) == 1:
            if trigger_times[i] - current_cluster[0] < boundarie_time:
                current_cluster.append(trigger_times[i])
            else:
                current_cluster.append(current_cluster[0])
                clusters.append(current_cluster)
                current_cluster = [trigger_times[i]]
        elif len(current_cluster) == 2:
            if trigger_times[i] - current_cluster[1] < boundarie_time:
                current_cluster[1] = trigger_times[i]
            else:
                clusters.append(current_cluster)
                current_cluster = [trigger_times[i]]
        i += 1
    if len(current_cluster) == 2:
        clusters.append(current_cluster)
    elif len(current_cluster) == 1:
        clusters.append([current_cluster[0], current_cluster[0]])
    return clusters



def load_data(path, epoch_offset=0., verbose=False, delta_t=0.1,
              data_activation='linear', target_activation='softmax'):
    if not os.path.isdir(path):
        raise ValueError('Path {} for loading data not found.'.format(path))
    files = os.listdir(path)
    out = []
    if verbose:
        bar = progress_tracker(len(files), name='Loading data')
    for fn in files:
        tmp_path = os.path.join(path, fn)
        if not os.path.isfile(tmp_path):
            if verbose:
                bar.iterate()
            continue
        try:
            with h5py.File(tmp_path, 'r') as fp:
                data = fp['data'][()]
                epoch = get_start_time(fn) + epoch_offset
                if data_activation == 'linear':
                    if target_activation == 'linear':
                        out.append(TimeSeries(data.T[0] - data.T[1],
                                              delta_t=delta_t,
                                              epoch=epoch))
                    elif target_activation == 'softmax':
                        exp0 = np.exp(data.T[0])
                        exp1 = np.exp(data.T[1])
                        ts = TimeSeries(exp0 / (exp0 + exp1),
                                        delta_t=delta_t,
                                        epoch=epoch)
                        out.append(ts)
                    else:
                        raise RuntimeError(f'Unrecognized target_activation {target_activation}.')
                elif data_activation == 'softmax':
                    if target_activation == 'softmax':
                        out.append(TimeSeries(data.T[0], delta_t=delta_t,
                                              epoch=epoch))
                    elif target_activation == 'linear':
                        raise ValueError('Cannot use target activation `linear` if data was generated with a softmax activation.')
                    else:
                        raise RuntimeError(f'Unrecognized target_activation {target_activation}.')
                else:
                    raise RuntimeError(f'Unrecognized data_activation {data_activation}.')
                if verbose:
                    bar.iterate()
        except:
            if verbose:
                bar.iterate()
            continue
    out = sorted(out, key=lambda ts: ts.start_time)
    return out

def assemble_time_series(ts_list):
    start = float(min(ts_list, key=lambda ts: ts.start_time).start_time)
    end = float(max(ts_list, key=lambda ts: ts.end_time).end_time)
    dts = {ts.delta_t for ts in ts_list}
    assert len(dts) == 1
    dt = dts.pop()
    ret = TimeSeries(np.zeros(int((end - start) / dt)+1),
                     delta_t=dt, epoch=start)
    assert int((end - float(ret.end_time)) / dt) == 0, 'Got end {} and end_time {} with {} samples difference.'.format(end, float(ret.end_time), int(abs(end - float(ret.end_time)) // dt))
    for ts in ts_list:
        start_idx = int(float(ts.start_time - ret.start_time) / dt)
        end_idx = start_idx + len(ts)
        ret.data[start_idx:end_idx] = ts.data[:]
    return ret

def custom_get_event_list_from_triggers(triggers, cluster_boundaries,
                                        verbose=False):
    events = []
    sort_idxs = np.argsort(triggers[0])
    sorted_triggers = (triggers.T[sort_idxs]).T
    if verbose:
        bar = progress_tracker(len(cluster_boundaries),
                               name='Calculating events')
    for cstart, cend in cluster_boundaries:
        sidx = np.searchsorted(sorted_triggers[0], cstart, side='left')
        eidx = np.searchsorted(sorted_triggers[0], cend, side='right')
        if sidx == eidx:
            logging.warn(f'Got a cluster that lies between two samples. Cluster: {(cstart, cend)}, Indices: {(sidx, eidx)}')
            continue
        idx = sidx + np.argmax(sorted_triggers[1][sidx:eidx])
        events.append((sorted_triggers[0][idx], sorted_triggers[1][idx]))
        if verbose:
            bar.iterate()
    return events

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--trigger-threshold', type=float, default=0.1,
                        help="The threshold value to determine triggers.")
    parser.add_argument('--cluster-tolerance', type=float, default=0.2,
                        help="The maximum distance (in seconds) between a trigger and the cluster boundaries for both to be clustered together.")
    parser.add_argument('--event-tolerance', type=float, default=0.3,
                        help="The maximum time (in seconds) between an event and an injection for them to be considered of the same origin.")
    parser.add_argument('--injection-file', required=True, type=str,
                        help="Path to the file containing the injections for this data.")
    parser.add_argument('--data-dir', type=str,
                        help="Path to the directory containing the output of the network. All files in this directory will be loaded.")
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing output files.')
    parser.add_argument('--delta-t', type=float, default=0.1,
                        help="The (actual) time (in seconds) between two slices. (By actual nsamples / sample_rate is meant)")
    parser.add_argument('--start-time-offset', type=float, default=0.75,
                        help="The time from the start of each processed window to the central point of the interval in which the merger time is varied.")
    parser.add_argument('--duration', type=float,
                        help="The duration of the data that is analyzed. Only required if triggers or events are loaded.")
    parser.add_argument('--test-data-activation', choices=['linear', 'softmax'], default='linear',
                        help="Which activation function was used to create the output. Default: `linear`")
    parser.add_argument('--ranking-statistic', choices=['softmax', 'linear'], default='softmax',
                        help="How should the output of the network be used to rate events? (This option may only be set to `linear`, if --test-data-activation is set to `linear`)")
    parser.add_argument('--verbose', action='store_true',
                        help="Print status updates.")
    parser.add_argument('--trigger-file-name', type=str, default='triggers.hdf',
                        help="The name of the trigger file that is stored in the --data-dir.")
    parser.add_argument('--event-file-name', type=str, default='events.hdf',
                        help="The name of the event file that is stored in the --data-dir.")
    parser.add_argument('--stats-file-name', type=str, default='statistics.hdf',
                        help="The name of the statistics file that is stored in the --data-dir.")
    parser.add_argument('--load-triggers', type=str,
                        help="Start analysis from the given trigger-file. (Argument must be the path to the file)")
    parser.add_argument('--load-events', type=str,
                        help="Start analysis from the given event-file. (Argument must be the path to the file)")
    
    args = parser.parse_args()
    
    log_level = logging.INFO if args.verbose else logging.WARN
    
    logging.basicConfig(format='%(levelname)s | %(asctime)s: %(message)s',
                        level=log_level, datefmt='%d-%m-%Y %H:%M:%S')
    
    logging.info('Started evaluation process')
    
    if args.ranking_statistic == 'linear':
        if args.test_data_activation != 'linear':
            raise ValueError(f'Can only use a linear ranking statistic if the test data was produced from a linear activation.')
    
    if args.load_triggers is None and args.load_events is None:
        if args.data_dir is None:
            raise ValueError(f'Must provide a directory from which to load the data when no triggers or events are loaded.')
        logging.info('Starting to load data')
        data = load_data(args.data_dir, epoch_offset=args.start_time_offset,
                        verbose=args.verbose, delta_t=args.delta_t,
                        data_activation=args.test_data_activation,
                        target_activation=args.ranking_statistic)
        logging.info(f'Loading complete. Loaded {len(data)} files.')
        logging.info('Assembling total timeseries')
        ts = assemble_time_series(data)
        logging.info('Assembling complete')
        if args.duration is None:
            args.duration = ts.duration
    else:
        if args.data_dir is None:
            args.data_dir = '.'
        if args.duration is None:
            raise ValueError(f'Duration required if data is not loaded.')
    
    if args.load_triggers is None and args.load_events is None:
        #Calculate triggers
        logging.info('Calculating triggers')
        triggers = get_triggers(ts, args.trigger_threshold)
        logging.info('Found {} triggers'.format(len(triggers[0])))

        #Write triggers to file
        trigger_path = os.path.join(args.data_dir,
                                    args.trigger_file_name)
        if os.path.isfile(trigger_path):
            if not args.force:
                msg = 'Cannot overwrite trigger file at {}. Set the flag '
                msg += '--force if you want to overwrite the file anyways.'
                msg = msg.format(trigger_path)
                raise IOError(msg)
        with h5py.File(trigger_path, 'w') as fp:
            fp.create_dataset('data', data=triggers[0])
            fp.create_dataset('trigger_values', data=triggers[1])
        logging.info("Wrote triggers to {}.".format(trigger_path))
    elif args.load_triggers is not None:
        with h5py.File(args.load_triggers, 'r') as fp:
            triggers = np.vstack([fp['data'][()],
                                  fp['trigger_values']])
        logging.info(f"Loaded {len(triggers[0])} triggers from {args.load_triggers}")
    
    if args.load_events is None:
        #Calculate events
        logging.info('Calculating cluster boundaries')
        cb = get_cluster_boundaries(triggers,
                                    boundarie_time=args.cluster_tolerance)
        logging.info('Found {} clusters.'.format(len(cb)))
        
        logging.info('Calculating events')
        events = custom_get_event_list_from_triggers(triggers, cb,
                                                    verbose=args.verbose)
        logging.info('Found {} events.'.format(len(events)))
        
        #Write events to file
        event_path = os.path.join(args.data_dir,
                                args.event_file_name)
        if os.path.isfile(event_path):
            if not args.force:
                msg = 'Cannot overwrite event file at {}. Set the flag '
                msg += '--force if you want to overwrite the file anyways.'
                msg = msg.format(event_path)
                raise IOError(msg)
        with h5py.File(event_path, 'w') as fp:
            fp.create_dataset('times', data=np.array(get_event_times(events)))
            fp.create_dataset('values', data=np.array([event[1] for event in events]))
        logging.info("Wrote events to {}.".format(event_path))
    else:
        with h5py.File(args.load_events, 'r') as fp:
            events = np.vstack([fp['times'][()],
                                fp['values'][()]])
        events = [tuple(pt) for pt in events.T]
    
    #Read injection file
    with h5py.File(args.injection_file, 'r') as fp:
        inj_times = fp['tc'][()]
        inj_idxs = np.arange(len(inj_times))
        mass1 = fp['mass1'][()]
        mass2 = fp['mass2'][()]
        dist = fp['distance'][()]
    
    #Calculate sensitivities and false-alarm rates
    logging.info('Splitting all events into true- and false-positives.')
    tp, fp = split_true_and_false_positives(events, inj_times,
                                            tolerance=args.event_tolerance)
    logging.info(f'Found {len(tp)} true and {len(fp)} false positives')
    logging.info(f'Sorting true and false positives by their ranking statistic')
    tp = np.array(sorted(tp, key=lambda inp: inp[1]))
    fp = np.array(sorted(fp, key=lambda inp: inp[1]))

    tptimes, tpvals = tp.T

    rank = []
    far = []
    sens_frac = []
    tidxs = []
    if args.verbose:
        bar = progress_tracker(len(fp), name='Calculating ranking steps and false-alarm rate')
    for i, event in enumerate(fp):
        t, val = event
        if len(rank) == 0:
            rank.append(val)
            far.append((len(fp) - i) / args.duration * SEC_PER_MONTH)
            tidx = np.searchsorted(tpvals, val, side='right')
            sens_frac.append(1 - float(tidx) / len(tp))
            tidxs.append(tidx)
            if args.verbose:
                bar.iterate()
            continue
        if val < rank[-1]:
            raise RuntimeError
        if rank[-1] == val:
            far[-1] = (len(fp) - i - 1) / args.duration * SEC_PER_MONTH
        else:
            rank.append(val)
            far.append((len(fp) - i - 1) / args.duration * SEC_PER_MONTH)
            tidx = np.searchsorted(tpvals, val, side='right')
            sens_frac.append(1 - float(tidx) / len(tp))
            tidxs.append(tidx)
        if args.verbose:
            bar.iterate()

    logging.info(f'Getting base-level found and missed indices')
    _, base_fidxs = get_closest_injection_times(inj_times, tptimes,
                                                        return_indices=True)

    logging.info(f'Starting to calculate sensitive volumes')
    #Calculations based on pycbc.sensitivity.volume_montecarlo
    max_distance = dist.max()
    mc_vtot = (4. / 3.) * np.pi * max_distance**3.
    Ninj = len(dist)
    mc_norm = float(Ninj)
    mc_prefactor = mc_vtot / mc_norm
    sens_vol = []
    sens_vol_err = []
    sens_dist = []
    if args.verbose:
        bar = progress_tracker(len(tidxs), name='Calculating sensitive volume')
    for idx in tidxs:
        mc_sum = len(base_fidxs) - idx
        mc_sample_variance = mc_sum / Ninj - (mc_sum / Ninj) ** 2
        vol = mc_prefactor * mc_sum
        vol_err = mc_prefactor * (Ninj * mc_sample_variance) ** 0.5
        
        sens_vol.append(vol)
        sens_vol_err.append(vol_err)
        rad = (3 * vol / (4 * np.pi))**(1. / 3.)
        sens_dist.append(rad)
        if args.verbose:
            bar.iterate()
    
    #Write FAR and sensitivity to file
    stat_path = os.path.join(args.data_dir, args.stats_file_name)
    if os.path.isfile(stat_path):
        if not args.force:
            msg = 'Cannot overwrite statistics file at {}. Set the flag '
            msg += '--force if you want to overwrite the file anyways.'
            msg = msg.format(stat_path)
            raise IOError(msg)
    with h5py.File(stat_path, 'w') as fp:
        fp.create_dataset('ranking', data=np.array(rank))
        fp.create_dataset('far', data=np.array(far))
        fp.create_dataset('sens-frac', data=np.array(sens_frac))
        fp.create_dataset('sens-dist', data=np.array(sens_dist))
        fp.create_dataset('sens-vol', data=np.array(sens_vol))
        fp.create_dataset('sens-vol-err', data=np.array(sens_vol_err))
    logging.info("Wrote statistics to {}.".format(stat_path))
    
    logging.info('Finished')
    return

if __name__ == "__main__":
    main()
