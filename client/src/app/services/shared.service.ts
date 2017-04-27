import { Injectable } from '@angular/core'
import { Observable } from 'rxjs';
import { Subject }    from 'rxjs/Subject';
import { ISearchResults } from '../entities/search-results.interface.js'

@Injectable()
export class SharedService {
    
    constructor() { }

    private searchResultsSource = new Subject<ISearchResults>();
    private searchTermSource = new Subject<string>();
    private isUserLoggedInSource = new Subject<boolean>();

    searchResult$ = this.searchResultsSource.asObservable();
    searchterm$ = this.searchTermSource.asObservable();
    isUserLoggedIn$ = this.isUserLoggedInSource.asObservable();

    addSearchResults(searchResults: ISearchResults) {
        this.searchResultsSource.next(searchResults)
    }

    addSearchTerm(searchTerm: string) {
        this.searchTermSource.next(searchTerm);
    }

    setLoginStatus(isUserLoggedIn: boolean) {
        this.isUserLoggedInSource.next(isUserLoggedIn);
    }
}